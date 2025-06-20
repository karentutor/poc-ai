# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from .renamer import rename_pdf
from fastapi.staticfiles import StaticFiles
import io, uuid, logging
from fastapi.responses import RedirectResponse

app = FastAPI(title="PDF Widget Renamer")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/rename")
async def rename(request: Request, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(415, "Only PDF files are accepted")
    data = await file.read()
    try:
        out_bin, mapping_info = rename_pdf(data)
    except Exception as e:
        logging.exception("rename failed")
        raise HTTPException(500, f"Rename failed: {e}")

    # Check the accept header from the request, not from the file
    accept_header = request.headers.get("accept", "")
    if accept_header and "application/json" in accept_header:
        return mapping_info

    return StreamingResponse(
        io.BytesIO(out_bin),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{uuid.uuid4()}.pdf"'
        },
    )

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")