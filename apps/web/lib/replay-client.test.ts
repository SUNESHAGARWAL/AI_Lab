import { describe, expect, it, vi } from "vitest";

import { cachedFixtureFor, replayCachedQuery } from "./replay-client";
import type { GraphEvent } from "./types/graph-events.generated";

describe("cachedFixtureFor", () => {
  it("resolves the real curated example questions", () => {
    const fixture = cachedFixtureFor("How does the GDPR define personal data?");
    expect(fixture).toBeDefined();
    expect(fixture![0]!.type).toBe("graph_started");
  });

  it("returns undefined for anything not an exact curated question", () => {
    expect(cachedFixtureFor("some random free-typed question")).toBeUndefined();
  });
});

describe("replayCachedQuery", () => {
  it("never touches the network — the whole point of the cached path", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const fixture = cachedFixtureFor("How does the GDPR define personal data?")!;

    const events: GraphEvent[] = [];
    for await (const event of replayCachedQuery(fixture)) events.push(event);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(events).toEqual(fixture);
    fetchSpy.mockRestore();
  });

  it("stops yielding once the signal is aborted", async () => {
    const fixture = cachedFixtureFor("How does the GDPR define personal data?")!;
    const controller = new AbortController();

    const events: GraphEvent[] = [];
    for await (const event of replayCachedQuery(fixture, controller.signal)) {
      events.push(event);
      if (events.length === 1) controller.abort();
    }

    expect(events.length).toBeLessThan(fixture.length);
  });
});
