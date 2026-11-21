"""FastAPI instance #2: convert old-district populations using a package."""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .batch import RULE_PREPARATIONS_HEADER, BatchInputError, BatchRunner
from .engine import ConversionError, convert
from .models import ConvertRequest, ConvertResponse, ErrorResponse

app = FastAPI(
    title="District Re-zoning Population Converter",
    description=(
        "Submit a handoff package (from the builder instance) and the "
        "currently available old-district populations (non-negative "
        "integers). Returns the new-district populations that can be "
        "derived exactly, together with their fractional-coefficient "
        "formulas; every other new district is marked as missing data. "
        "A contradicted coverage relation, or a negative/non-integer "
        "result, makes the whole request fail with an error response.\n\n"
        "POST /convert/batch additionally converts a whole historical "
        "table in one call: upload the v1 handoff package and a UTF-8 CSV "
        "(first column row_id, then one column per old district; blank = "
        "not reported, 0 = reported). The converted CSV is downloaded "
        "after the whole batch passes."
    ),
    version="1.1.0",
)


@app.exception_handler(ConversionError)
def _conversion_error_handler(_request, exc: ConversionError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="conversion_failed", detail=str(exc)).model_dump(),
    )


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "population-converter",
        "docs": "/docs",
        "endpoints": ["/convert", "/convert/batch"],
    }


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post(
    "/convert",
    response_model=ConvertResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["convert"],
    summary="Convert known old populations into new-district populations",
)
def run_convert(request: ConvertRequest) -> ConvertResponse:
    result = convert(
        package=request.package.model_dump(),
        old_populations=dict(request.old_populations),
    )
    return ConvertResponse(**result)


@app.post(
    "/convert/batch",
    tags=["convert"],
    summary="Convert a whole historical table (CSV) to the new districts",
    responses={
        200: {
            "description": (
                "The converted table as a CSV download. Columns: row_id, "
                "then every new district in package order; cells that "
                "cannot be derived are left blank. The "
                "X-Rule-Preparations response header reports how often "
                "conversion rules were prepared (once per distinct "
                "reported-district combination)."
            ),
            "content": {"text/csv": {}},
        },
        400: {"model": ErrorResponse},
    },
)
def run_convert_batch(
    package: UploadFile = File(
        ..., description="v1 handoff package (JSON) from the builder instance"
    ),
    table: UploadFile = File(
        ...,
        description=(
            "UTF-8 CSV: first column row_id, then exactly one column per "
            "old district of the package; blank = not reported, 0 = reported"
        ),
    ),
):
    runner = BatchRunner()
    try:
        return runner.run(package, table)
    except BatchInputError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="invalid_input", detail=str(exc)
            ).model_dump(),
            headers={RULE_PREPARATIONS_HEADER: str(runner.preparations)},
        )
    except ConversionError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="conversion_failed", detail=str(exc)
            ).model_dump(),
            headers={RULE_PREPARATIONS_HEADER: str(runner.preparations)},
        )
