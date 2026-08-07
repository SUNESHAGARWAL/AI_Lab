"""Dumps the JSON Schema for api.graph.events.GraphEvent to stdout.

Source of truth is the Pydantic model, not this script — see apps/web/package.json's
`generate:types` script, which pipes this output through json-schema-to-typescript into
apps/web/lib/types/graph-events.generated.ts. Re-run via `pnpm --filter web generate:types`
whenever api/graph/events.py changes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "apps/api/src"))

from pydantic import TypeAdapter  # noqa: E402

from api.graph.events import GraphEvent  # noqa: E402


def main() -> None:
    schema = TypeAdapter(GraphEvent).json_schema()
    schema["title"] = "GraphEvent"
    # Pydantic marks `type` as not-required because it has a default — but every event
    # is always serialized with it, and json-schema-to-typescript needs it required to
    # emit a proper TS discriminated union (narrowing on an optional field is unreliable).
    for definition in schema["$defs"].values():
        if "type" in definition.get("properties", {}):
            definition.setdefault("required", [])
            if "type" not in definition["required"]:
                definition["required"].append("type")
    json.dump(schema, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
