from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from io import BytesIO
import shutil
import os
import json
from PIL import Image
import requests
import time
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from random import randint
import uuid
from pathlib import Path



IMAGEDIR = "images/"
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# Directory where HTML files are stored
templates_dir = Path(__file__).parent / "templates"

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open(templates_dir / "index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/beautify", response_class=HTMLResponse)
async def get_beautify():
    with open(templates_dir / "beautify.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/sketch", response_class=HTMLResponse)
async def get_sketch():
    with open(templates_dir / "sketch.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/complete", response_class=HTMLResponse)
async def get_complete():
    with open(templates_dir / "complete.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/generate", response_class=HTMLResponse)
async def get_generate():
    with open(templates_dir / "generate.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/search", response_class=HTMLResponse)
async def get_search():
    with open(templates_dir / "search.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpeg"
    contents = await file.read()
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
    return {"filename": file.filename}

@app.get("/show/")
async def read_random_file():
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)
    path = f"{IMAGEDIR}{files[random_index]}"
    return FileResponse(path)

STABILITY_KEY = 'sk-1ejsB6TPWJ7IqeqTTWm5qDyjfSoDRiEhIx1hoa5gbO6lb9Bv'

class Sd3(BaseModel):
    prompt: str
    negative_prompt: str = ""
    aspect_ratio: str = "1:1"
    output_format: str = "jpeg"
    model: str = "sd3"
    seed: int = randint(0, 2 ** 32 - 1)

async def send_generation_request(host, params):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response


async def send_async_generation_request(host, params):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    files = {}
    if "image" in params:
        image = params.pop("image")
        files = {"image": open(image, 'rb')}

    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    response_dict = response.json()
    generation_id = response_dict.get("id", None)
    assert generation_id is not None, "Expected id in response"

    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    status_code = 202
    while status_code == 202:
        response = requests.get(
            f"{host}/result/{generation_id}",
            headers={
                **headers,
                "Accept": "image/*"
            },
        )

        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        status_code = response.status_code
        time.sleep(10)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

    return response


def clear_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)


@app.post("/generate_from_text/")
async def generate_from_text(sd3: Sd3):
    params = {
        "prompt": sd3.prompt,
        "negative_prompt": sd3.negative_prompt,
        "aspect_ratio": sd3.aspect_ratio,
        "seed": sd3.seed,
        "output_format": sd3.output_format,
        "model": sd3.model,
        "mode": "text-to-image"
    }
    host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"
    try:
        response = await send_generation_request(host, params)
        output_image = Image.open(BytesIO(response.content))
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")

        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")

        clear_directory("images")

        generated = f"generated_{seed}.{sd3.output_format}"

        os.makedirs("images", exist_ok=True)
        full_path = os.path.join("images", generated)
        output_image.save(full_path)
        return FileResponse(full_path, media_type=f"image/{sd3.output_format}")
    except json.JSONDecodeError as err:
        return JSONResponse(status_code=500, content={"error": str(err)})

@app.post("/generate_from_image_and_text/")
async def generate_from_image_and_text(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        control_strength: float = Form(0.7),
        seed: int = Form(randint(0, 2 ** 32 - 1)),
        output_format: str = Form("jpeg")
):
    try:
        image_path = f"{IMAGEDIR}{uuid.uuid4()}.jpeg"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        params = {
            "control_strength": control_strength,
            "image": image_path,
            "seed": seed,
            "output_format": output_format,
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }
        host = f"https://api.stability.ai/v2beta/stable-image/control/sketch"
        response = await send_generation_request(host, params)
        output_image = Image.open(BytesIO(response.content))
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")
        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")
        clear_directory("images")
        edited = f"edited_{seed}.{output_format}"
        full_path = os.path.join("images", edited)
        output_image.save(full_path)
        return FileResponse(full_path, media_type=f"image/{output_format}")
    except json.JSONDecodeError as err:
        return JSONResponse(status_code=500, content={"error": str(err)})
    except Warning as warn:
        return JSONResponse(status_code=400, content={"warning": str(warn)})

@app.post("/generate_image_beautify/")
async def generate_image_beautify(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        control_strength: float = Form(0.7),
        seed: int = Form(randint(0, 2 ** 32 - 1)),
        output_format: str = Form("jpeg")
):
    try:
        image_path = f"{IMAGEDIR}{uuid.uuid4()}.jpeg"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        params = {
            "control_strength": control_strength,
            "image": image_path,
            "seed": seed,
            "output_format": output_format,
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }
        host = f"https://api.stability.ai/v2beta/stable-image/control/sketch"
        response = await send_generation_request(host, params)
        if response.status_code != 200:
            return JSONResponse(status_code=response.status_code, content={"error": response.text})
        output_image = response.content
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")
        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")
        filename, _ = os.path.splitext(os.path.basename(image_path))
        edited = f"edited_{filename}_{seed}.{output_format}"
        edited_path = os.path.join(IMAGEDIR, edited)
        with open(edited_path, "wb") as f:
            f.write(output_image)
        return FileResponse(edited_path, media_type=f"image/{output_format}")
    except json.JSONDecodeError as err:
        return JSONResponse(status_code=500, content={"error": str(err)})
    except Warning as warn:
        return JSONResponse(status_code=400, content={"warning": str(warn)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate_complete_image/")
async def generate_complete_image(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        left: int = Form(800),
        right: int = Form(800),
        up: int = Form(200),
        down: int = Form(1536),
        creativity: float = Form(0.5),
        seed: int = Form(randint(0, 2 ** 32 - 1)),
        output_format: str = Form("jpeg")
):
    try:
        image_path = f"{IMAGEDIR}{uuid.uuid4()}.jpeg"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        params = {
            "image": image_path,
            "left": left,
            "right": right,
            "up": up,
            "down": down,
            "prompt": prompt,
            "creativity": creativity,
            "seed": seed,
            "output_format": output_format
        }
        host = f"https://api.stability.ai/v2beta/stable-image/edit/outpaint"
        response = await send_generation_request(host, params)
        output_image = Image.open(BytesIO(response.content))
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")
        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")
        clear_directory("images")
        edited = f"edited_{seed}.{output_format}"
        full_path = os.path.join("images", edited)
        output_image.save(full_path)
        return FileResponse(full_path, media_type=f"image/{output_format}")
    except json.JSONDecodeError as err:
        return JSONResponse(status_code=500, content={"error": str(err)})
    except Warning as warn:
        return JSONResponse(status_code=400, content={"warning": str(warn)})

@app.post("/generate_search_replace_image/")
async def generate_search_replace_image(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        search_prompt: str = Form(...),
        negative_prompt: str = Form(""),
        seed: int = Form(randint(0, 2 ** 32 - 1)),
        output_format: str = Form("jpeg")
):
    try:
        # Save the uploaded image
        image_path = f"{IMAGEDIR}{uuid.uuid4()}.jpeg"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        params = {
            "image": image_path,
            "seed": seed,
            "mode": "search",
            "output_format": output_format,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "search_prompt": search_prompt,
        }
        host = f"https://api.stability.ai/v2beta/stable-image/edit/search-and-replace"

        response = await send_generation_request(host, params)
        output_image = Image.open(BytesIO(response.content))
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")

        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")

        clear_directory("images")

        edited = f"edited_{seed}.{output_format}"
        full_path = os.path.join("images", edited)
        output_image.save(full_path)

        return FileResponse(full_path, media_type=f"image/{output_format}")
    except json.JSONDecodeError as err:
        return JSONResponse(status_code=500, content={"error": str(err)})
    except Warning as warn:
        return JSONResponse(status_code=400, content={"warning": str(warn)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9000)
