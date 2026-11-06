def plotly_pdf_bytes(payload: dict) -> bytes:
    from copy import deepcopy

    import plotly.io as pio

    export_payload = deepcopy(payload)
    for trace in export_payload.get("data", []):
        if trace.get("type") == "scattergl":
            trace["type"] = "scatter"

    layout = export_payload.get("layout", {})
    return pio.to_image(
        export_payload,
        format="pdf",
        width=max(1, round(layout.get("width", 1800) * 0.6)),
        height=layout.get("height", 450),
        scale=1,
    )


def as_plotly_payload(figure: object) -> dict:
    if figure is None:
        raise TypeError(
            "The loaded plot() function returned None. Restart the Streamlit process "
            "to discard the older Matplotlib implementation."
        )
    serializer = getattr(figure, "to_plotly_json", None)
    if serializer is None:
        raise TypeError(f"plot() returned {type(figure).__name__}, not a Plotly figure")
    payload = serializer()
    if not isinstance(payload, dict):
        raise TypeError("The Plotly figure serializer did not return a dictionary")
    return payload
