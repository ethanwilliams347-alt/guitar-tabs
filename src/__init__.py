"""video_to_sheet pipeline: ingest -> canvas -> rows -> notes -> read -> render."""


class Refused(Exception):
    """The video breaks an assumption of the pipeline. The message is the reason."""
