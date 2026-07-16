# Face API Testing

This guide covers every public API in `face-classifier-service` with curl commands
and complete Python/Go test scripts. Examples assume:

- `BASE_URL=http://127.0.0.1:8000`
- optional `API_BASE_PATH` is already included in `BASE_URL`, for example `http://127.0.0.1:8000/face-api`
- `TOKEN=primary-token` when `FACE_AUTH_ENABLED=true`
- `IMAGE=C:\path\face.jpg`

If auth is disabled, remove the `Authorization` header. For the native model
runner, configure:

```env
FACE_ANALYZER_FACTORY=face_classifier_service.native_analyzer:create_analyzer
```

If `/readyz` returns `503`, model artifacts or the analyzer factory are not ready.
The commands are still valid, but model endpoints will return `503` until the
native analyzer loads.

## curl

PowerShell setup:

```powershell
$BASE_URL = "http://127.0.0.1:8000"
$TOKEN = "primary-token"
$IMAGE = "C:\path\face.jpg"
$AUTH = "Authorization: Bearer $TOKEN"
```

Health and readiness:

```powershell
curl.exe -H $AUTH "$BASE_URL/healthz"
curl.exe -H $AUTH "$BASE_URL/readyz"
```

Encryption public key:

```powershell
curl.exe -H $AUTH "$BASE_URL/v1/encryption/public-key"
```

Plain multipart face APIs:

```powershell
curl.exe -H $AUTH -F "files=@$IMAGE" "$BASE_URL/v1/faces/detect"
curl.exe -H $AUTH -F "files=@$IMAGE" "$BASE_URL/v1/faces/align?include_landmarks106=true&include_aligned_crop=false"
curl.exe -H $AUTH -F "files=@$IMAGE" "$BASE_URL/v1/faces/quality?include_landmarks106=true"
curl.exe -H $AUTH -F "files=@$IMAGE" "$BASE_URL/v1/faces/embed"
curl.exe -H $AUTH -F "files=@$IMAGE" "$BASE_URL/v1/faces/analyze?include_landmarks106=true&include_aligned_crop=false"
```

To request debug JSON embedding output, start the service with
`FACE_ALLOW_DEBUG_EMBEDDING_JSON=true`:

```powershell
curl.exe -H $AUTH -F "files=@$IMAGE" "$BASE_URL/v1/faces/embed?embedding_format=json"
curl.exe -H $AUTH -F "files=@$IMAGE" "$BASE_URL/v1/faces/analyze?embedding_format=json"
```

Generate one encrypted JSON payload with Python:

```powershell
@'
import base64, json, os, sys, urllib.request
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nacl.public import PublicKey, SealedBox

base_url, token, image_path, output_path = sys.argv[1:5]
headers = {"Authorization": f"Bearer {token}"} if token else {}
request = urllib.request.Request(
    base_url.rstrip("/") + "/v1/encryption/public-key",
    headers=headers,
    method="GET",
)
with urllib.request.urlopen(request, timeout=15) as response:
    public_key = json.loads(response.read().decode("utf-8"))["public_key"]

image_bytes = Path(image_path).read_bytes()
aes_key = os.urandom(32)
nonce = os.urandom(12)
ciphertext = AESGCM(aes_key).encrypt(nonce, image_bytes, None)
encrypted_key = SealedBox(PublicKey(base64.b64decode(public_key))).encrypt(aes_key)
payload = {
    "items": [{
        "filename": Path(image_path).name,
        "alg": "AES-256-GCM+SealedBox-v1",
        "encrypted_key": base64.b64encode(encrypted_key).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }]
}
Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
Path(output_path + ".key").write_text(base64.b64encode(aes_key).decode("ascii"), encoding="ascii")
'@ | Set-Content .\encrypt_face_payload.py

.\.venv\Scripts\python.exe .\encrypt_face_payload.py $BASE_URL $TOKEN $IMAGE .\encrypted-face.json
```

Encrypted face APIs. These return an encrypted response envelope instead of
plaintext face data:
`{"alg":"AES-256-GCM+SealedBox-v1","key_index":0,"nonce":"...","ciphertext":"..."}`.
`key_index` selects which uploaded item AES key decrypts the response; for a
single-image request this is the `.key` file generated above. Face boxes,
landmarks, alignment data, quality scores, and embeddings are all inside the
encrypted response plaintext.

```powershell
curl.exe -H $AUTH -H "Content-Type: application/json" --data-binary "@encrypted-face.json" "$BASE_URL/v1/faces/detect/encrypted"
curl.exe -H $AUTH -H "Content-Type: application/json" --data-binary "@encrypted-face.json" "$BASE_URL/v1/faces/align/encrypted?include_landmarks106=true&include_aligned_crop=false"
curl.exe -H $AUTH -H "Content-Type: application/json" --data-binary "@encrypted-face.json" "$BASE_URL/v1/faces/quality/encrypted?include_landmarks106=true"
curl.exe -H $AUTH -H "Content-Type: application/json" --data-binary "@encrypted-face.json" "$BASE_URL/v1/faces/embed/encrypted"
curl.exe -H $AUTH -H "Content-Type: application/json" --data-binary "@encrypted-face.json" "$BASE_URL/v1/faces/analyze/encrypted?include_landmarks106=true&include_aligned_crop=false"
```

Decrypt one encrypted response saved from curl:

```powershell
curl.exe -H $AUTH -H "Content-Type: application/json" --data-binary "@encrypted-face.json" "$BASE_URL/v1/faces/analyze/encrypted" -o encrypted-response.json

@'
import base64, json, sys
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

response_path, key_path = sys.argv[1:3]
encrypted = json.loads(Path(response_path).read_text(encoding="utf-8"))
aes_key = base64.b64decode(Path(key_path).read_text(encoding="ascii"))
nonce = base64.b64decode(encrypted["nonce"])
ciphertext = base64.b64decode(encrypted["ciphertext"])
print(AESGCM(aes_key).decrypt(nonce, ciphertext, None).decode("utf-8"))
'@ | Set-Content .\decrypt_face_response.py

.\.venv\Scripts\python.exe .\decrypt_face_response.py .\encrypted-response.json .\encrypted-face.json.key
```

## Python Script

Install client dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install requests pynacl cryptography
```

Save as `face_api_test.py`:

```python
import argparse
import base64
import json
import os
from pathlib import Path

import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nacl.public import PublicKey, SealedBox


ENDPOINTS = ["detect", "align", "quality", "embed", "analyze"]
ALG = "AES-256-GCM+SealedBox-v1"


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_public_key(base_url: str, token: str | None) -> str:
    response = requests.get(
        f"{base_url.rstrip('/')}/v1/encryption/public-key",
        headers=auth_headers(token),
        timeout=15,
    )
    print("GET /v1/encryption/public-key", response.status_code, response.text)
    response.raise_for_status()
    return response.json()["public_key"]


def encrypt_image(public_key_b64: str, image_path: Path) -> tuple[dict, bytes]:
    image_bytes = image_path.read_bytes()
    aes_key = os.urandom(32)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, image_bytes, None)
    encrypted_key = SealedBox(PublicKey(base64.b64decode(public_key_b64))).encrypt(aes_key)
    return (
        {
            "filename": image_path.name,
            "alg": ALG,
            "encrypted_key": base64.b64encode(encrypted_key).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        },
        aes_key,
    )


def decrypt_response(encrypted_response: dict, aes_key: bytes) -> dict:
    if encrypted_response.get("alg") != ALG:
        raise ValueError("unsupported response encryption algorithm")
    nonce = base64.b64decode(encrypted_response["nonce"])
    ciphertext = base64.b64decode(encrypted_response["ciphertext"])
    plaintext = AESGCM(aes_key).decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))


def query(endpoint: str, json_embedding: bool) -> dict[str, str]:
    params: dict[str, str] = {}
    if endpoint in {"align", "quality", "analyze"}:
        params["include_landmarks106"] = "true"
    if endpoint in {"align", "analyze"}:
        params["include_aligned_crop"] = "false"
    if json_embedding and endpoint in {"embed", "analyze"}:
        params["embedding_format"] = "json"
    return params


def call_plain(base_url: str, token: str | None, endpoint: str, image_path: Path, json_embedding: bool) -> None:
    with image_path.open("rb") as file:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/faces/{endpoint}",
            headers=auth_headers(token),
            params=query(endpoint, json_embedding),
            files=[("files", (image_path.name, file, "application/octet-stream"))],
            timeout=60,
        )
    print(f"POST /v1/faces/{endpoint}", response.status_code, response.text)


def call_encrypted(
    base_url: str,
    token: str | None,
    endpoint: str,
    payload: dict,
    aes_key: bytes,
    json_embedding: bool,
) -> None:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/faces/{endpoint}/encrypted",
        headers={**auth_headers(token), "Content-Type": "application/json"},
        params=query(endpoint, json_embedding),
        json={"items": [payload]},
        timeout=60,
    )
    print(f"POST /v1/faces/{endpoint}/encrypted", response.status_code, response.text)
    if response.ok:
        print("decrypted response", json.dumps(decrypt_response(response.json(), aes_key), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token")
    parser.add_argument("--image", required=True)
    parser.add_argument("--endpoint", choices=[*ENDPOINTS, "all"], default="all")
    parser.add_argument("--encrypted", action="store_true")
    parser.add_argument("--json-embedding", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    image_path = Path(args.image)
    endpoints = ENDPOINTS if args.endpoint == "all" else [args.endpoint]

    for path in ["/healthz", "/readyz"]:
        response = requests.get(base_url + path, headers=auth_headers(args.token), timeout=15)
        print("GET", path, response.status_code, response.text)

    payload = None
    aes_key = None
    if args.encrypted:
        payload, aes_key = encrypt_image(get_public_key(base_url, args.token), image_path)

    for endpoint in endpoints:
        if args.encrypted:
            call_encrypted(base_url, args.token, endpoint, payload, aes_key, args.json_embedding)
        else:
            call_plain(base_url, args.token, endpoint, image_path, args.json_embedding)


if __name__ == "__main__":
    main()
```

Run all plaintext APIs:

```powershell
.\.venv\Scripts\python.exe .\face_api_test.py --base-url $BASE_URL --token $TOKEN --image $IMAGE --endpoint all
```

Run all encrypted APIs:

```powershell
.\.venv\Scripts\python.exe .\face_api_test.py --base-url $BASE_URL --token $TOKEN --image $IMAGE --endpoint all --encrypted
```

Run one API:

```powershell
.\.venv\Scripts\python.exe .\face_api_test.py --base-url $BASE_URL --token $TOKEN --image $IMAGE --endpoint analyze --encrypted --json-embedding
```

## Go Script

Initialize once:

```powershell
go mod init face-api-test
go get golang.org/x/crypto/nacl/box
```

Save as `face_api_client.go`:

```go
package main

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/nacl/box"
)

const alg = "AES-256-GCM+SealedBox-v1"

var endpoints = []string{"detect", "align", "quality", "embed", "analyze"}

type PublicKeyResponse struct {
	PublicKey string `json:"public_key"`
}

type EncryptedImagePayload struct {
	Filename     string `json:"filename,omitempty"`
	Alg          string `json:"alg"`
	EncryptedKey string `json:"encrypted_key"`
	Nonce        string `json:"nonce"`
	Ciphertext   string `json:"ciphertext"`
}

type EncryptedBatchPayload struct {
	Items []EncryptedImagePayload `json:"items"`
}

type EncryptedUpload struct {
	Payload EncryptedImagePayload
	AESKey  []byte
}

type EncryptedResponse struct {
	Alg        string `json:"alg"`
	KeyIndex   int    `json:"key_index"`
	Nonce      string `json:"nonce"`
	Ciphertext string `json:"ciphertext"`
}

func addAuth(req *http.Request, token string) {
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

func do(req *http.Request) {
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Println(req.Method, req.URL.String(), "ERROR", err)
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	fmt.Println(req.Method, req.URL.String(), resp.StatusCode, string(body))
}

func getPublicKey(baseURL, token string) (string, error) {
	req, _ := http.NewRequest("GET", strings.TrimRight(baseURL, "/")+"/v1/encryption/public-key", nil)
	addAuth(req, token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	fmt.Println("GET", req.URL.String(), resp.StatusCode, string(body))
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("public key request failed: %s", resp.Status)
	}
	var parsed PublicKeyResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", err
	}
	return parsed.PublicKey, nil
}

func encryptImage(publicKeyB64, imagePath string) (EncryptedUpload, error) {
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		return EncryptedUpload{}, err
	}
	publicKeyBytes, err := base64.StdEncoding.DecodeString(publicKeyB64)
	if err != nil {
		return EncryptedUpload{}, err
	}
	if len(publicKeyBytes) != 32 {
		return EncryptedUpload{}, fmt.Errorf("public key must be 32 bytes")
	}
	var publicKey [32]byte
	copy(publicKey[:], publicKeyBytes)

	aesKey := make([]byte, 32)
	if _, err := rand.Read(aesKey); err != nil {
		return EncryptedUpload{}, err
	}
	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return EncryptedUpload{}, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return EncryptedUpload{}, err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return EncryptedUpload{}, err
	}
	ciphertext := gcm.Seal(nil, nonce, imageBytes, nil)
	encryptedKey, err := box.SealAnonymous(nil, aesKey, &publicKey, rand.Reader)
	if err != nil {
		return EncryptedUpload{}, err
	}
	return EncryptedUpload{
		Payload: EncryptedImagePayload{
			Filename:     filepath.Base(imagePath),
			Alg:          alg,
			EncryptedKey: base64.StdEncoding.EncodeToString(encryptedKey),
			Nonce:        base64.StdEncoding.EncodeToString(nonce),
			Ciphertext:   base64.StdEncoding.EncodeToString(ciphertext),
		},
		AESKey: aesKey,
	}, nil
}

func decryptResponse(body []byte, aesKey []byte) ([]byte, error) {
	var encrypted EncryptedResponse
	if err := json.Unmarshal(body, &encrypted); err != nil {
		return nil, err
	}
	if encrypted.Alg != alg {
		return nil, fmt.Errorf("unsupported response encryption algorithm")
	}
	nonce, err := base64.StdEncoding.DecodeString(encrypted.Nonce)
	if err != nil {
		return nil, err
	}
	ciphertext, err := base64.StdEncoding.DecodeString(encrypted.Ciphertext)
	if err != nil {
		return nil, err
	}
	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return gcm.Open(nil, nonce, ciphertext, nil)
}

func params(endpoint string, jsonEmbedding bool) string {
	values := url.Values{}
	if endpoint == "align" || endpoint == "quality" || endpoint == "analyze" {
		values.Set("include_landmarks106", "true")
	}
	if endpoint == "align" || endpoint == "analyze" {
		values.Set("include_aligned_crop", "false")
	}
	if jsonEmbedding && (endpoint == "embed" || endpoint == "analyze") {
		values.Set("embedding_format", "json")
	}
	encoded := values.Encode()
	if encoded == "" {
		return ""
	}
	return "?" + encoded
}

func callPlain(baseURL, token, endpoint, imagePath string, jsonEmbedding bool) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", filepath.Base(imagePath))
	if err != nil {
		fmt.Println(err)
		return
	}
	file, err := os.Open(imagePath)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer file.Close()
	io.Copy(part, file)
	writer.Close()

	req, _ := http.NewRequest(
		"POST",
		strings.TrimRight(baseURL, "/")+"/v1/faces/"+endpoint+params(endpoint, jsonEmbedding),
		&body,
	)
	addAuth(req, token)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	do(req)
}

func callEncrypted(baseURL, token, endpoint string, upload EncryptedUpload, jsonEmbedding bool) {
	body, _ := json.Marshal(EncryptedBatchPayload{Items: []EncryptedImagePayload{upload.Payload}})
	req, _ := http.NewRequest(
		"POST",
		strings.TrimRight(baseURL, "/")+"/v1/faces/"+endpoint+"/encrypted"+params(endpoint, jsonEmbedding),
		bytes.NewReader(body),
	)
	addAuth(req, token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Println(req.Method, req.URL.String(), "ERROR", err)
		return
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	fmt.Println(req.Method, req.URL.String(), resp.StatusCode, string(respBody))
	if resp.StatusCode < 300 {
		decrypted, err := decryptResponse(respBody, upload.AESKey)
		if err != nil {
			fmt.Println("decrypt response ERROR", err)
			return
		}
		fmt.Println("decrypted response", string(decrypted))
	}
}

func main() {
	baseURL := flag.String("base-url", "http://127.0.0.1:8000", "")
	token := flag.String("token", "", "")
	image := flag.String("image", "", "")
	endpoint := flag.String("endpoint", "all", "detect|align|quality|embed|analyze|all")
	encrypted := flag.Bool("encrypted", false, "")
	jsonEmbedding := flag.Bool("json-embedding", false, "")
	flag.Parse()
	if *image == "" {
		panic("--image is required")
	}

	for _, path := range []string{"/healthz", "/readyz"} {
		req, _ := http.NewRequest("GET", strings.TrimRight(*baseURL, "/")+path, nil)
		addAuth(req, *token)
		do(req)
	}

	selected := endpoints
	if *endpoint != "all" {
		selected = []string{*endpoint}
	}

	var upload EncryptedUpload
	if *encrypted {
		publicKey, err := getPublicKey(*baseURL, *token)
		if err != nil {
			panic(err)
		}
		upload, err = encryptImage(publicKey, *image)
		if err != nil {
			panic(err)
		}
	}

	for _, ep := range selected {
		if *encrypted {
			callEncrypted(*baseURL, *token, ep, upload, *jsonEmbedding)
		} else {
			callPlain(*baseURL, *token, ep, *image, *jsonEmbedding)
		}
	}
}
```

Run all plaintext APIs:

```powershell
go run .\face_api_client.go --base-url "$BASE_URL" --token "$TOKEN" --image "$IMAGE" --endpoint all
```

Run all encrypted APIs:

```powershell
go run .\face_api_client.go --base-url "$BASE_URL" --token "$TOKEN" --image "$IMAGE" --endpoint all --encrypted
```

Run one API:

```powershell
go run .\face_api_client.go --base-url "$BASE_URL" --token "$TOKEN" --image "$IMAGE" --endpoint analyze --encrypted --json-embedding
```

## Expected Errors

- `401 missing bearer token`: add `Authorization: Bearer <token>`.
- `401 invalid bearer token`: token is absent from `FACE_API_TOKENS`.
- `403 encrypted uploads are required`: use `/v1/faces/<endpoint>/encrypted`.
- `403 json embedding output requires allow_debug_embedding_json=true`: set `FACE_ALLOW_DEBUG_EMBEDDING_JSON=true` or omit `embedding_format=json`.
- `404 encrypted uploads are not enabled`: set `FACE_ENCRYPTION_ENABLED=true` and configure key files.
- `400 invalid encrypted payload`: payload is malformed, encrypted with a stale public key, or tampered.
- `413 request body too large`: request exceeded plaintext or encrypted body limits.
- `503 model is not ready`: model artifacts or `FACE_ANALYZER_FACTORY` are not ready.
