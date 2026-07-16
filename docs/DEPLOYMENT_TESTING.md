# Face Service Deployment Testing

This file targets the local deployment at:

- deploy dir: `C:\Users\Admin\deploy\face-classifier-service`
- nginx URL: `http://127.0.0.1:8080/face`
- direct service URL: `http://127.0.0.1:51993/face`
- token file: `C:\Users\Admin\deploy\face-classifier-service\config\api_token.txt`

`FACE_REQUIRE_ENCRYPTED_UPLOADS=true` is enabled. Plain face endpoints should
return `403`; encrypted face endpoints return encrypted envelopes. The current
native model runner is:

```env
FACE_ANALYZER_FACTORY=face_classifier_service.native_analyzer:create_analyzer
```

## Setup

PowerShell variables:

```powershell
$DEPLOY = "$HOME\deploy\face-classifier-service"
$BASE_URL = "http://127.0.0.1:8080/face"
$DIRECT_URL = "http://127.0.0.1:51993/face"
$TOKEN = (Get-Content "$DEPLOY\config\api_token.txt" -Raw).Trim()
$AUTH = "Authorization: Bearer $TOKEN"
$IMAGE = "$DEPLOY\test-assets\face-test.png"
```

Create a small smoke-test image if you do not have a real face image ready:

```powershell
$code = @'
from pathlib import Path
from PIL import Image
p = Path(r"__IMAGE__")
p.parent.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (64, 64), (220, 80, 70)).save(p)
print(p)
'@
$code = $code.Replace("__IMAGE__", $IMAGE.Replace("\", "\\"))
$code | & "$DEPLOY\.venv\Scripts\python.exe" -
```

## curl Cases

Service and nginx status:

```powershell
Get-Service -Name OpenAIFaceClassifierService,nginx |
  Select-Object Name,Status,StartType

Get-NetTCPConnection -LocalPort 51993 -State Listen |
  Select-Object LocalAddress,LocalPort,State,OwningProcess
```

Health, auth, docs, and readiness:

```powershell
curl.exe -s -o - -w "`nhealth_direct=%{http_code}`n" -H $AUTH "$DIRECT_URL/healthz"
curl.exe -s -o - -w "`nhealth_nginx=%{http_code}`n" -H $AUTH "$BASE_URL/healthz"
curl.exe -s -o - -w "`nmissing_auth=%{http_code}`n" "$BASE_URL/healthz"
curl.exe -s -o - -w "`ninvalid_auth=%{http_code}`n" -H "Authorization: Bearer wrong" "$BASE_URL/healthz"
curl.exe -s -o - -w "`npublic_key=%{http_code}`n" -H $AUTH "$BASE_URL/v1/encryption/public-key"
curl.exe -s -o - -w "`nreadyz=%{http_code}`n" -H $AUTH "$BASE_URL/readyz"
curl.exe -s -o - -w "`ndocs=%{http_code}`n" -H $AUTH "$BASE_URL/docs"
```

Expected:

- health direct/nginx: `200`
- missing/invalid auth: `401`
- public key: `200`
- docs: `404`
- readyz: `200`

Plain face APIs should be rejected because encrypted uploads are required:

```powershell
foreach ($ep in "detect","align","quality","embed","analyze") {
  curl.exe -s -o - -w "`nplain_$ep=%{http_code}`n" `
    -H $AUTH `
    -F "files=@$IMAGE" `
    "$BASE_URL/v1/faces/$ep"
}
```

Expected: `403 {"detail":"encrypted uploads are required"}` for all five.

Generate one encrypted upload payload:

```powershell
@'
import base64, json, os, sys, urllib.request
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nacl.public import PublicKey, SealedBox

base_url, token, image_path, output_path = sys.argv[1:5]
request = urllib.request.Request(
    base_url.rstrip("/") + "/v1/encryption/public-key",
    headers={"Authorization": f"Bearer {token}"},
    method="GET",
)
with urllib.request.urlopen(request, timeout=15) as response:
    public_key = json.loads(response.read().decode("utf-8"))["public_key"]

image = Path(image_path)
aes_key = os.urandom(32)
nonce = os.urandom(12)
ciphertext = AESGCM(aes_key).encrypt(nonce, image.read_bytes(), None)
encrypted_key = SealedBox(PublicKey(base64.b64decode(public_key))).encrypt(aes_key)
payload = {
    "items": [{
        "filename": image.name,
        "alg": "AES-256-GCM+SealedBox-v1",
        "encrypted_key": base64.b64encode(encrypted_key).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }]
}
Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
Path(output_path + ".key").write_text(base64.b64encode(aes_key).decode("ascii"), encoding="ascii")
'@ | Set-Content "$DEPLOY\test-assets\make_encrypted_payload.py"

& "$DEPLOY\.venv\Scripts\python.exe" `
  "$DEPLOY\test-assets\make_encrypted_payload.py" `
  $BASE_URL $TOKEN $IMAGE "$DEPLOY\test-assets\encrypted-face.json"
```

Call every encrypted face API:

```powershell
foreach ($ep in "detect","align","quality","embed","analyze") {
  curl.exe -s -o "$DEPLOY\test-assets\encrypted-$ep-response.json" -w "encrypted_$ep=%{http_code}`n" `
    -H $AUTH `
    -H "Content-Type: application/json" `
    --data-binary "@$DEPLOY\test-assets\encrypted-face.json" `
    "$BASE_URL/v1/faces/$ep/encrypted"
}
```

Expected: `200` and each response body should be an encrypted envelope with only
`alg`, `key_index`, `nonce`, and `ciphertext`. Face boxes, landmarks, quality
scores, aligned crops, and embeddings are inside the encrypted plaintext after
client-side decryption.

Decrypt a successful encrypted response:

```powershell
@'
import base64, json, sys
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

response_path, key_path = sys.argv[1:3]
encrypted = json.loads(Path(response_path).read_text(encoding="utf-8"))
aes_key = base64.b64decode(Path(key_path).read_text(encoding="ascii"))
plaintext = AESGCM(aes_key).decrypt(
    base64.b64decode(encrypted["nonce"]),
    base64.b64decode(encrypted["ciphertext"]),
    None,
)
print(plaintext.decode("utf-8"))
'@ | Set-Content "$DEPLOY\test-assets\decrypt_response.py"

& "$DEPLOY\.venv\Scripts\python.exe" `
  "$DEPLOY\test-assets\decrypt_response.py" `
  "$DEPLOY\test-assets\encrypted-analyze-response.json" `
  "$DEPLOY\test-assets\encrypted-face.json.key"
```

## Python Full Runner

Save as `$DEPLOY\test-assets\face_deploy_test.py`:

```python
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nacl.public import PublicKey, SealedBox
from PIL import Image


ENDPOINTS = ["detect", "align", "quality", "embed", "analyze"]
ALG = "AES-256-GCM+SealedBox-v1"


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.new("RGB", (64, 64), (220, 80, 70)).save(path)


def encrypt_image(public_key_b64: str, image_path: Path) -> tuple[dict, bytes]:
    aes_key = os.urandom(32)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, image_path.read_bytes(), None)
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


def decrypt_response(payload: dict, aes_key: bytes) -> dict:
    if payload.get("alg") != ALG:
        raise AssertionError(f"unexpected response alg: {payload.get('alg')}")
    plaintext = AESGCM(aes_key).decrypt(
        base64.b64decode(payload["nonce"]),
        base64.b64decode(payload["ciphertext"]),
        None,
    )
    return json.loads(plaintext.decode("utf-8"))


def expect(status: int, allowed: set[int], label: str, body: str) -> None:
    print(label, status, body)
    if status not in allowed:
        raise AssertionError(f"{label}: expected {allowed}, got {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", default=str(Path.home() / "deploy" / "face-classifier-service"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/face")
    parser.add_argument("--image")
    parser.add_argument("--expect-ready", action="store_true")
    args = parser.parse_args()

    deploy = Path(args.deploy)
    token = (deploy / "config" / "api_token.txt").read_text(encoding="ascii").strip()
    image = Path(args.image) if args.image else deploy / "test-assets" / "face-test.png"
    make_image(image)
    headers = {"Authorization": f"Bearer {token}"}

    for path in ["/healthz", "/v1/encryption/public-key"]:
        response = requests.get(args.base_url.rstrip("/") + path, headers=headers, timeout=15)
        expect(response.status_code, {200}, f"GET {path}", response.text)

    ready = requests.get(args.base_url.rstrip("/") + "/readyz", headers=headers, timeout=15)
    expect(ready.status_code, {200} if args.expect_ready else {503, 200}, "GET /readyz", ready.text)

    missing = requests.get(args.base_url.rstrip("/") + "/healthz", timeout=15)
    expect(missing.status_code, {401}, "GET /healthz without token", missing.text)

    for endpoint in ENDPOINTS:
        with image.open("rb") as file:
            response = requests.post(
                args.base_url.rstrip("/") + f"/v1/faces/{endpoint}",
                headers=headers,
                files=[("files", (image.name, file, "image/png"))],
                timeout=30,
            )
        expect(response.status_code, {403}, f"POST plain {endpoint}", response.text)

    public_key = requests.get(
        args.base_url.rstrip("/") + "/v1/encryption/public-key",
        headers=headers,
        timeout=15,
    ).json()["public_key"]

    encrypted, aes_key = encrypt_image(public_key, image)
    allowed_encrypted = {200} if args.expect_ready else {503, 200}
    for endpoint in ENDPOINTS:
        response = requests.post(
            args.base_url.rstrip("/") + f"/v1/faces/{endpoint}/encrypted",
            headers={**headers, "Content-Type": "application/json"},
            json={"items": [encrypted]},
            timeout=60,
        )
        expect(response.status_code, allowed_encrypted, f"POST encrypted {endpoint}", response.text)
        if response.status_code == 200:
            envelope = response.json()
            if "results" in envelope:
                raise AssertionError("encrypted endpoint leaked plaintext results")
            decoded = decrypt_response(envelope, aes_key)
            if "results" not in decoded:
                raise AssertionError("decrypted response has no results")
            print("decrypted", endpoint, json.dumps(decoded, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

Run the deployment test suite. Use a real face image for meaningful face outputs;
without `--image`, the generated smoke image should still return `200` with
`faces: []`.

```powershell
& "$DEPLOY\.venv\Scripts\python.exe" "$DEPLOY\test-assets\face_deploy_test.py" --expect-ready --image "C:\path\real-face.jpg"
```

## Go Full Runner

Initialize once in a temporary client directory:

```powershell
mkdir "$DEPLOY\test-assets\go-client" -Force | Out-Null
cd "$DEPLOY\test-assets\go-client"
go mod init face-deploy-test
go get golang.org/x/crypto/nacl/box
```

Save as `$DEPLOY\test-assets\go-client\main.go`:

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
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/nacl/box"
)

const alg = "AES-256-GCM+SealedBox-v1"

var endpoints = []string{"detect", "align", "quality", "embed", "analyze"}

type publicKeyResponse struct {
	PublicKey string `json:"public_key"`
}

type encryptedImagePayload struct {
	Filename     string `json:"filename,omitempty"`
	Alg          string `json:"alg"`
	EncryptedKey string `json:"encrypted_key"`
	Nonce        string `json:"nonce"`
	Ciphertext   string `json:"ciphertext"`
}

type encryptedBatchPayload struct {
	Items []encryptedImagePayload `json:"items"`
}

type encryptedResponse struct {
	Alg        string `json:"alg"`
	KeyIndex   int    `json:"key_index"`
	Nonce      string `json:"nonce"`
	Ciphertext string `json:"ciphertext"`
}

func addAuth(req *http.Request, token string) {
	req.Header.Set("Authorization", "Bearer "+token)
}

func check(label string, status int, allowed map[int]bool, body []byte) {
	fmt.Println(label, status, string(body))
	if !allowed[status] {
		panic(fmt.Sprintf("%s: unexpected status %d", label, status))
	}
}

func request(method, url, token string, body io.Reader, contentType string) (int, []byte) {
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		panic(err)
	}
	if token != "" {
		addAuth(req, token)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, respBody
}

func getPublicKey(baseURL, token string) string {
	status, body := request("GET", strings.TrimRight(baseURL, "/")+"/v1/encryption/public-key", token, nil, "")
	check("public key", status, map[int]bool{200: true}, body)
	var parsed publicKeyResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		panic(err)
	}
	return parsed.PublicKey
}

func encryptImage(publicKeyB64, imagePath string) (encryptedImagePayload, []byte) {
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		panic(err)
	}
	publicKeyBytes, err := base64.StdEncoding.DecodeString(publicKeyB64)
	if err != nil {
		panic(err)
	}
	var publicKey [32]byte
	copy(publicKey[:], publicKeyBytes)
	aesKey := make([]byte, 32)
	rand.Read(aesKey)
	block, _ := aes.NewCipher(aesKey)
	gcm, _ := cipher.NewGCM(block)
	nonce := make([]byte, gcm.NonceSize())
	rand.Read(nonce)
	ciphertext := gcm.Seal(nil, nonce, imageBytes, nil)
	encryptedKey, err := box.SealAnonymous(nil, aesKey, &publicKey, rand.Reader)
	if err != nil {
		panic(err)
	}
	return encryptedImagePayload{
		Filename:     filepath.Base(imagePath),
		Alg:          alg,
		EncryptedKey: base64.StdEncoding.EncodeToString(encryptedKey),
		Nonce:        base64.StdEncoding.EncodeToString(nonce),
		Ciphertext:   base64.StdEncoding.EncodeToString(ciphertext),
	}, aesKey
}

func decryptResponse(body []byte, aesKey []byte) []byte {
	var encrypted encryptedResponse
	if err := json.Unmarshal(body, &encrypted); err != nil {
		panic(err)
	}
	if encrypted.Alg != alg {
		panic("unexpected response alg")
	}
	nonce, _ := base64.StdEncoding.DecodeString(encrypted.Nonce)
	ciphertext, _ := base64.StdEncoding.DecodeString(encrypted.Ciphertext)
	block, _ := aes.NewCipher(aesKey)
	gcm, _ := cipher.NewGCM(block)
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		panic(err)
	}
	return plaintext
}

func callPlain(baseURL, token, endpoint, imagePath string) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, _ := writer.CreateFormFile("files", filepath.Base(imagePath))
	file, err := os.Open(imagePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	io.Copy(part, file)
	writer.Close()
	status, body := request(
		"POST",
		strings.TrimRight(baseURL, "/")+"/v1/faces/"+endpoint,
		token,
		&buf,
		writer.FormDataContentType(),
	)
	check("plain "+endpoint, status, map[int]bool{403: true}, body)
}

func callEncrypted(baseURL, token, endpoint string, payload encryptedImagePayload, aesKey []byte, expectReady bool) {
	body, _ := json.Marshal(encryptedBatchPayload{Items: []encryptedImagePayload{payload}})
	status, respBody := request(
		"POST",
		strings.TrimRight(baseURL, "/")+"/v1/faces/"+endpoint+"/encrypted",
		token,
		bytes.NewReader(body),
		"application/json",
	)
	allowed := map[int]bool{200: true, 503: !expectReady}
	check("encrypted "+endpoint, status, allowed, respBody)
	if status == 200 {
		fmt.Println("decrypted "+endpoint, string(decryptResponse(respBody, aesKey)))
	}
}

func main() {
	baseURL := flag.String("base-url", "http://127.0.0.1:8080/face", "")
	deploy := flag.String("deploy", filepath.Join(os.Getenv("USERPROFILE"), "deploy", "face-classifier-service"), "")
	image := flag.String("image", "", "")
	expectReady := flag.Bool("expect-ready", false, "")
	flag.Parse()

	tokenBytes, err := os.ReadFile(filepath.Join(*deploy, "config", "api_token.txt"))
	if err != nil {
		panic(err)
	}
	token := strings.TrimSpace(string(tokenBytes))
	imagePath := *image
	if imagePath == "" {
		imagePath = filepath.Join(*deploy, "test-assets", "face-test.png")
	}

	status, body := request("GET", strings.TrimRight(*baseURL, "/")+"/healthz", token, nil, "")
	check("health", status, map[int]bool{200: true}, body)
	status, body = request("GET", strings.TrimRight(*baseURL, "/")+"/readyz", token, nil, "")
	check("readyz", status, map[int]bool{200: true, 503: !*expectReady}, body)
	status, body = request("GET", strings.TrimRight(*baseURL, "/")+"/healthz", "", nil, "")
	check("missing auth", status, map[int]bool{401: true}, body)

	for _, endpoint := range endpoints {
		callPlain(*baseURL, token, endpoint, imagePath)
	}
	publicKey := getPublicKey(*baseURL, token)
	payload, aesKey := encryptImage(publicKey, imagePath)
	for _, endpoint := range endpoints {
		callEncrypted(*baseURL, token, endpoint, payload, aesKey, *expectReady)
	}
}
```

Run the Go deployment test suite:

```powershell
cd "$DEPLOY\test-assets\go-client"
go run . --base-url "$BASE_URL" --deploy "$DEPLOY" --image "C:\path\real-face.jpg" --expect-ready
```

## Model Artifact Checks

Verify all model files are inside the deployment directory and match their
sidecars. This includes CVLFace `model.safetensors`, local wrapper/config, and
`pretrained_model/model.pt`, because the native runtime reads those files at
startup:

```powershell
cd $DEPLOY
& ".\.venv\Scripts\python.exe" -c "from face_classifier_service.settings import Settings; from face_classifier_service.models import validate_model_artifacts; s=Settings(); print(s.models_dir); validate_model_artifacts(s.models_dir); print('model_artifacts=valid')"
Get-Content "$DEPLOY\models\MODEL_SOURCES.json"
```

## Analyzer Factory Check

Verify the factory setting and native runtime devices:

```powershell
cd $DEPLOY
Select-String -Path ".env" -Pattern "^FACE_ANALYZER_FACTORY="
& ".\.venv\Scripts\python.exe" -c "from face_classifier_service.settings import Settings; from face_classifier_service.native_analyzer import create_analyzer; s=Settings(); a=create_analyzer(s); print('ready=', a.ready); print('detector=', a.detector.session.get_providers()); print('landmark=', a.landmark_model.session.get_providers()); print('quality=', a.quality_model.device); print('embedder=', a.embedder.device)"
```
