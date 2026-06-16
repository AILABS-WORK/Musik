"""AudioSet-527 tagging (AST). See ``audioset.py``."""

from mgc.tagging.audioset import (
    AudioSetTagger,
    get_audioset_labels,
    tag_all,
    top_tags,
)

__all__ = ["AudioSetTagger", "get_audioset_labels", "tag_all", "top_tags"]
