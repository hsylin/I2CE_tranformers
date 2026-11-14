(() => {
  "use strict";

  function decodeEvidence(payload) {
    if (payload.encoding !== "columnar-v1") return payload;
    const objectFromRow = (fields, values) =>
      Object.fromEntries(fields.map((field, index) => [field, values[index]]));
    const points = Object.fromEntries(payload.rows.map(([id, values, phases]) => [
      id, {id, ...objectFromRow(payload.point_fields, values),
        phases: Object.fromEntries(phases.map(([name, metrics]) =>
          [name, objectFromRow(payload.phase_fields, metrics)]))}
    ]));
    return {...payload.metadata, points};
  }

  const evidence = decodeEvidence(JSON.parse(document.getElementById("tile-evidence-data").textContent));
  const node = id => document.getElementById(id);
  const escape = value => String(value).replace(/[&<>"']/g, char =>
    ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
  const tuple = point => evidence.axes.map(axis => point.tiles[axis]).join("/");
  const namedTiles = point => evidence.axes
    .map(axis => axis + ": " + point.tiles[axis]).join(" / ");
  const pair = (point, first, second) => point.tiles[first] + "/" + point.tiles[second];
  const numericPairOrder = (a, b) => {
    const left = a.split("/").map(Number), right = b.split("/").map(Number);
    return left[0] - right[0] || left[1] - right[1];
  };
  const uniquePairs = (points, first, second) =>
    [...new Set(points.map(point => pair(point, first, second)))].sort(numericPairOrder);
  const milliseconds = ticks => (ticks / 1e9).toFixed(3);

  function belongsToProfile(point, profile) {
    return point && profile && ["cache", "learners", "entries"].every(key => point[key] === profile[key]);
  }

  function validMeasurement(point, profile) {
    return belongsToProfile(point, profile) && point.writeback_clean === false &&
      typeof point.binary_sha256 === "string" && point.binary_sha256.length > 0 &&
      Number.isSafeInteger(point.BLOCK_ticks) && point.BLOCK_ticks > 0 &&
      evidence.axes.every(axis => Number.isSafeInteger(point.tiles?.[axis]) && point.tiles[axis] > 0);
  }

  function profileView(profile) {
    if (!profile) return {profile: null, points: [], cells: new Map(), minimum: null};
    const candidates = [...new Set(profile.grid_ids)]
      .map(id => evidence.points[id]).filter(point => validMeasurement(point, profile));
    const grouped = new Map();
    candidates.forEach(point => {
      const key = tuple(point);
      grouped.set(key, [...(grouped.get(key) || []), point]);
    });
    // Ambiguous duplicate geometries are not silently replaced by the fastest row.
    const points = [...grouped.values()].filter(group => group.length === 1).map(group => group[0]);
    const comparable = new Set(points.map(point => point.binary_sha256)).size === 1;
    return {profile, points, cells: new Map(points.map(point => [tuple(point), point])),
      minimum: comparable && points.length ? Math.min(...points.map(point => point.BLOCK_ticks)) : null};
  }

  function colorFor(gap, maximum) {
    const stops = [[255,255,204],[255,237,160],[254,217,118],[254,178,76],
      [253,141,60],[252,78,42],[227,26,28],[189,0,38],[128,0,38]];
    const position = Math.max(0, Math.min(1, gap / maximum)) * (stops.length - 1);
    const low = Math.min(Math.floor(position), stops.length - 2), weight = position - low;
    const rgb = stops[low].map((value, index) =>
      Math.round(value + weight * (stops[low + 1][index] - value)));
    return {background: "rgb(" + rgb.join(",") + ")", foreground: position >= 5.5 ? "#fff" : "#111"};
  }

  function cellHtml(point, view, selected, maximum, invalid) {
    if (invalid) return '<td class="missing"><span class="cell invalid" title="Invalid: O1 > O2">Invalid: O1 &gt; O2</span></td>';
    if (!point) return '<td class="missing"><span class="cell" title="Not measured in the original grid">—</span></td>';
    const gap = view.minimum === null ? null : 100 * (point.BLOCK_ticks / view.minimum - 1);
    const color = gap === null ? {background: "#eef0f2", foreground: "#111"} : colorFor(gap, maximum);
    const best = view.minimum !== null && point.BLOCK_ticks === view.minimum;
    const formula = selected && tuple(selected) === tuple(point);
    const label = point.id + "; tiles s1/o1/k1/s2/o2/k2 = " + tuple(point) +
      "; BLOCK " + milliseconds(point.BLOCK_ticks) + " ms" +
      (gap === null ? "; different binaries: no ranking" : "; " + gap.toFixed(3) + "% above cache best") +
      (best ? "; observed best" : "") + (formula ? "; frozen formula geometry" : "");
    const classes = "cell" + (best ? " best" : "") + (formula ? " formula" : "");
    return '<td data-point-id="' + escape(point.id) + '" style="background:' + color.background +
      ";color:" + color.foreground + '"><span tabindex="0" class="' + classes +
      '" title="' + escape(label) + '" aria-label="' + escape(label) + '">' +
      milliseconds(point.BLOCK_ticks) + "</span></td>";
  }

  function selectionNote(view, selected) {
    if (!view.profile) return '<p class="selection-note">Missing profile.</p>';
    if (!selected) return '<p class="selection-note">Frozen formula: Not measured.</p>';
    if (view.cells.has(tuple(selected))) return "";
    return '<p class="selection-note"><strong>Frozen formula</strong> <code>' +
      escape(namedTiles(selected)) + '</code> · <span class="formula-time">' +
      milliseconds(selected.BLOCK_ticks) + " ms</span></p>";
  }

  function cacheHtml(cache, view, axes, maximum) {
    const sizes = evidence.cache_bytes[cache];
    const selectedPoint = view.profile && evidence.points[view.profile.references.selected];
    const selected = validMeasurement(selectedPoint, view.profile) ? selectedPoint : null;
    const headers = axes.columns.map(column => '<th scope="col">' + column + "</th>").join("");
    const bodies = axes.rows.map(row => "<tbody>" + axes.inner.map((inner, index) => {
      const [s1, s2] = row.split("/").map(Number), [o1, k1] = inner.split("/").map(Number);
      const cells = axes.columns.map(column => {
        const [o2, k2] = column.split("/").map(Number);
        const key = evidence.axes.map(axis => ({s1, o1, k1, s2, o2, k2})[axis]).join("/");
        return cellHtml(view.cells.get(key), view, selected, maximum, o1 > o2);
      }).join("");
      return "<tr>" + (index === 0 ? '<th class="row-group" scope="rowgroup" rowspan="' +
        axes.inner.length + '">' + row + "</th>" : "") +
        '<th class="inner-label" scope="row">' + inner + "</th>" + cells + "</tr>";
    }).join("") + "</tbody>").join("");
    return '<article class="cache-panel" data-cache="' + cache + '"><h3>Cache ' + cache +
      " — L1D " + sizes[0] / 1024 + " KiB / L2 " + sizes[1] / 1048576 + " MiB</h3>" +
      '<div class="heat-scroll"><table class="grid-heatmap" aria-label="Cache ' +
      cache + ' BLOCK runtime"><colgroup><col class="axis-col"><col class="axis-col">' +
      axes.columns.map(() => "<col>").join("") + "</colgroup><thead><tr>" +
      '<th rowspan="2" scope="col">S1 / S2</th><th rowspan="2" scope="col">O1 / K1</th><th colspan="' +
      axes.columns.length + '" scope="colgroup">O2 / K2</th></tr><tr>' + headers +
      "</tr></thead>" + bodies + "</table></div>" + selectionNote(view, selected) + "</article>";
  }

  function refresh() {
    const learners = Number(node("tile-learners").value);
    const entries = Number(node("tile-entries").value);
    const profiles = evidence.profiles.filter(profile => profile.learners === learners && profile.entries === entries);
    const views = ["A", "B", "C", "D"].map(cache => profileView(profiles.find(profile => profile.cache === cache)));
    const available = views.flatMap(view => view.points);
    // Use the frozen grid geometry for common axes, even when a profile is missing.
    const allGrid = evidence.profiles.flatMap(profile => profile.grid_ids)
      .map(id => evidence.points[id]).filter(point => point && point.tiles);
    const axisPoints = allGrid.length ? allGrid : available;
    const axes = {rows: uniquePairs(axisPoints, "s1", "s2"),
      inner: uniquePairs(axisPoints, "o1", "k1"), columns: uniquePairs(axisPoints, "o2", "k2")};
    const gaps = views.flatMap(view => view.minimum === null ? [] :
      view.points.map(point => 100 * (point.BLOCK_ticks / view.minimum - 1)));
    const maximum = Math.max(1, Math.ceil(Math.max(0, ...gaps)));
    node("grid-scale-ticks").innerHTML = [0, maximum / 2, maximum]
      .map(value => "<span>" + Number(value.toFixed(1)) + "%</span>").join("");
    node("grid-cache-panels").innerHTML = views.map((view, index) =>
      cacheHtml(["A", "B", "C", "D"][index], view, axes, maximum)).join("");
  }

  ["tile-learners", "tile-entries"].forEach(id => node(id).addEventListener("change", refresh));
  // Native HTML tables do not depend on the containing tab's dimensions.
  refresh();
})();
