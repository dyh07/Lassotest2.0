const { useEffect, useMemo, useRef, useState } = React;

const SELECTION_COLORS = [
  "#ef4444",
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
  "#f97316",
  "#14b8a6",
];

const ENCODER_LAYER_OPTIONS = [
  [256, 128, 64],
  [256, 64],
  [64, 32],
  [64],
];

const DISCRIMINATOR_LAYER_OPTIONS = [
  [256, 64],
  [64, 32],
  [64],
];

const LAMBDA_ATTENTION_OPTIONS = [0.1, 0.2, 0.5, 1.0];

function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

function layersToKey(layers) {
  return layers.join("-");
}

function keyToLayers(key) {
  return key.split("-").map((value) => Number(value));
}

function quotePython(value) {
  if (value === null || value === undefined) {
    return "None";
  }
  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => quotePython(item)).join(", ")}]`;
  }
  return String(value);
}

function indentBlock(lines) {
  return lines.map((line) => `    ${line}`).join("\n");
}

function buildLassoViewCode({ activeSelection, summary, viewOneConfig, analysisConfig }) {
  const selectedIds = activeSelection?.ids || [];
  const obsCol = viewOneConfig.colorBy || summary?.default_color_by || "annotation";
  return [
    "from backend.do_lasso import do_lasso",
    "",
    `selected_ids = ${quotePython(selectedIds)}`,
    `obs_col = ${quotePython(obsCol)}`,
    "",
    "expanded_ids = do_lasso(",
    indentBlock([
      "adata=adata,",
      "user_selected_list=selected_ids,",
      `obs_col=${quotePython(obsCol)},`,
      "vis=False,",
      `vis_key=${quotePython(obsCol)},`,
      `do_correct=${quotePython(Boolean(analysisConfig.doCorrect))},`,
    ]),
    ")",
  ].join("\n");
}

function buildDownsampleCode({ summary, viewOneConfig, analysisConfig }) {
  const embeddingKey = viewOneConfig.embeddingKey || summary?.default_embedding || "X_umap";
  const colorBy = viewOneConfig.colorBy || summary?.default_color_by || "leiden";
  return [
    "from backend.do_downsample import do_h5ad_downsample",
    "",
    "downsampled_adata, nearest_ids = do_h5ad_downsample(",
    indentBlock([
      "adata=adata,",
      `sample_rate=${quotePython(Number(analysisConfig.sampleRate))},`,
      `leiden_r=${quotePython(Number(analysisConfig.leidenResolution))},`,
      `uniform_rate=${quotePython(Number(analysisConfig.uniformRate))},`,
      "add_col='orig_idx',",
      `cluster_key=${quotePython(colorBy)},`,
      `obsm_key=${quotePython(embeddingKey)},`,
    ]),
    ")",
  ].join("\n");
}

function createViewConfig(summary, plot, overrides = {}) {
  return {
    embeddingKey: plot?.embedding_key || summary?.default_embedding || "",
    colorBy: plot?.color_by || summary?.default_color_by || "",
    pointSize: 4.5,
    opacity: 0.82,
    invertX: false,
    invertY: false,
    ...overrides,
  };
}

function makeCategoryColors(values) {
  const fallback = "#90a4b8";
  if (!values) {
    return { colors: [], legend: [] };
  }

  const categories = [];
  const seen = new Set();
  values.forEach((value) => {
    const key = value === null || value === undefined ? "Unassigned" : String(value);
    if (!seen.has(key)) {
      seen.add(key);
      categories.push(key);
    }
  });

  const palette = categories.map((_, index) => {
    const hue = Math.round((index * 137.508) % 360);
    return `hsl(${hue}, 68%, 52%)`;
  });
  const lookup = new Map(categories.map((item, index) => [item, palette[index] || fallback]));

  return {
    colors: values.map((value) => lookup.get(value === null || value === undefined ? "Unassigned" : String(value)) || fallback),
    legend: categories.map((label) => ({ label, color: lookup.get(label) || fallback })),
  };
}

function arrayUnion(left, right) {
  const next = new Set(left);
  right.forEach((item) => next.add(item));
  return Array.from(next).sort((a, b) => a - b);
}

function selectionKindConfig(kind) {
  if (kind === "refining") {
    return {
      displayLabelPrefix: "Refining",
      variablePrefix: "refine_list",
    };
  }
  return {
    displayLabelPrefix: "Selection",
    variablePrefix: "select_list",
  };
}

function normalizeSelectionCatalog(selections) {
  const counters = new Map();
  return selections.map((selection, index) => {
    const kind = selection.kind || "selection";
    const count = (counters.get(kind) || 0) + 1;
    counters.set(kind, count);
    const kindMeta = selectionKindConfig(kind);
    return {
      ...selection,
      kind,
      displayName: `${kindMeta.displayLabelPrefix} ${count}`,
      variableName: `${kindMeta.variablePrefix}${count}`,
      color: selection.color || SELECTION_COLORS[index % SELECTION_COLORS.length],
    };
  });
}

function buildBaseColorArray(plotData, baseColors, selectedOnly) {
  const fallback = Array(plotData.points.ids.length).fill("#7aa3f0");
  const source = baseColors.length ? baseColors : fallback;
  return selectedOnly ? Array(plotData.points.ids.length).fill("#d8e0ea") : source;
}

function buildSubsetTrace(plotData, ids, color, pointSize, opacity, lineColor, lineWidth) {
  const xs = [];
  const ys = [];
  const customdata = [];
  ids.forEach((id) => {
    if (id < 0 || id >= plotData.points.x.length || id >= plotData.points.y.length) {
      return;
    }
    xs.push(plotData.points.x[id]);
    ys.push(plotData.points.y[id]);
    customdata.push(id);
  });

  return {
    x: xs,
    y: ys,
    type: "scattergl",
    mode: "markers",
    hovertemplate: "Cell ID: %{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
    customdata,
    marker: {
      size: pointSize,
      opacity,
      color,
      line: {
        color: lineColor,
        width: lineWidth,
      },
    },
  };
}

function buildPlotTraces({
  plotData,
  baseColors,
  viewConfig,
  interactive,
  selectedOnly,
  pendingIds,
  confirmedSelections,
}) {
  const baseTrace = {
    x: plotData.points.x,
    y: plotData.points.y,
    type: "scattergl",
    mode: "markers",
    hovertemplate: "Cell ID: %{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
    customdata: plotData.points.ids,
    marker: {
      size: viewConfig.pointSize,
      opacity: viewConfig.opacity,
      color: buildBaseColorArray(plotData, baseColors, !interactive && selectedOnly),
      line: {
        color: "rgba(255,255,255,0.7)",
        width: 0.5,
      },
    },
  };

  const traces = [baseTrace];

  if (!interactive) {
    confirmedSelections.forEach((selection) => {
      if (selection.ids.length) {
        traces.push(buildSubsetTrace(
          plotData,
          selection.ids,
          selection.color,
          Math.max(viewConfig.pointSize + 0.4, 5),
          0.96,
          "#ffffff",
          0.8,
        ));
      }
    });
  }

  if (pendingIds.length) {
    traces.push(buildSubsetTrace(
      plotData,
      pendingIds,
      interactive ? "#0f172a" : "#111827",
      Math.max(viewConfig.pointSize + 0.8, 5.4),
      0.98,
      "#ffffff",
      1.1,
    ));
  }

  return traces;
}

function SelectionTextModal({ open, title, text, copied, onClose, onCopy }) {
  if (!open) {
    return null;
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3>{title}</h3>
            <p>Copy the selected IDs in a Python-friendly format.</p>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>Close</button>
        </div>
        <textarea readOnly value={text} className="id-textarea" />
        <div className="modal-actions">
          <button type="button" onClick={onCopy}>Copy content</button>
          <button type="button" className="ghost-button" onClick={onClose}>Dismiss</button>
          {copied ? <span className="copy-hint">Copied to clipboard.</span> : null}
        </div>
      </div>
    </div>
  );
}

function SelectionTabs({
  selections,
  activeSelectionId,
  overlapCountMap,
  onSelect,
  onShowIds,
  onDelete,
}) {
  if (!selections.length) {
    return (
      <section className="selection-tabs panel-card">
        <div className="section-heading">
          <h3>Selection classes</h3>
          <p>No confirmed selection classes yet. Confirm a draft from View 1 to create one.</p>
        </div>
      </section>
    );
  }

  const activeSelection = selections.find((selection) => selection.id === activeSelectionId) || selections[selections.length - 1];

  return (
    <section className="selection-tabs panel-card">
      <div className="section-heading">
        <h3>Selection classes</h3>
        <p>Review each confirmed selection class, inspect its IDs, or remove it.</p>
      </div>
      <div className="selection-tab-row">
        {selections.map((selection, index) => (
          <button
            type="button"
            key={selection.id}
            className={classNames("selection-tab", activeSelection.id === selection.id && "selection-tab-active")}
            onClick={() => onSelect(selection.id)}
          >
            <span className="selection-tab-dot" style={{ backgroundColor: selection.color }} />
            <span>{selection.displayName || `Selection ${index + 1}`}</span>
          </button>
        ))}
      </div>
      <div className="selection-tab-panel">
        <div className="selection-tab-meta">
          <div>
            <span className="summary-label">Python variable</span>
            <strong>{activeSelection.variableName}</strong>
          </div>
          <div>
            <span className="summary-label">Cells in this class</span>
            <strong>{activeSelection.ids.length.toLocaleString()}</strong>
          </div>
          <div>
            <span className="summary-label">Overlaps in this class</span>
            <strong>{(overlapCountMap.get(activeSelection.id) || 0).toLocaleString()}</strong>
          </div>
        </div>
        <div className="selection-tab-actions">
          <button type="button" className="ghost-button" onClick={() => onShowIds(activeSelection)}>
            Show IDs
          </button>
          <button type="button" className="ghost-button danger-button" onClick={() => onDelete(activeSelection.id)}>
            Delete this selection
          </button>
        </div>
      </div>
    </section>
  );
}

function SettingsPanel({ title, open, onToggle, summary, config, onConfigChange, busy }) {
  return (
    <section className="side-card side-card-settings">
      <button type="button" className="section-toggle" onClick={onToggle}>
        <span>{title}</span>
        <span>{open ? "Hide" : "Open"}</span>
      </button>
      {open ? (
        <div className="settings-stack">
          <label>
            <span>Embedding</span>
            <select
              value={config.embeddingKey}
              onChange={(event) => onConfigChange({ embeddingKey: event.target.value })}
              disabled={!summary?.available_embeddings?.length || busy}
            >
              <option value="">Select an embedding</option>
              {(summary?.available_embeddings || []).map((key) => (
                <option value={key} key={key}>{key}</option>
              ))}
            </select>
          </label>

          <label>
            <span>Color by</span>
            <select
              value={config.colorBy}
              onChange={(event) => onConfigChange({ colorBy: event.target.value })}
              disabled={!summary?.obs_columns?.length || busy}
            >
              <option value="">No coloring</option>
              {(summary?.obs_columns || []).map((column) => (
                <option value={column} key={column}>{column}</option>
              ))}
            </select>
          </label>

          <label className="slider-block">
            <div className="slider-label">
              <span>Cell size</span>
              <strong>{config.pointSize.toFixed(1)}</strong>
            </div>
            <input
              type="range"
              min="2"
              max="16"
              step="0.5"
              value={config.pointSize}
              onChange={(event) => onConfigChange({ pointSize: Number(event.target.value) })}
            />
          </label>

          <label className="slider-block">
            <div className="slider-label">
              <span>Opacity</span>
              <strong>{config.opacity.toFixed(2)}</strong>
            </div>
            <input
              type="range"
              min="0.15"
              max="1"
              step="0.05"
              value={config.opacity}
              onChange={(event) => onConfigChange({ opacity: Number(event.target.value) })}
            />
          </label>

          <div className="check-grid">
            <label className="checkbox-line">
              <input
                type="checkbox"
                checked={config.invertX}
                onChange={(event) => onConfigChange({ invertX: event.target.checked })}
              />
              <span>Inverse X axis</span>
            </label>
            <label className="checkbox-line">
              <input
                type="checkbox"
                checked={config.invertY}
                onChange={(event) => onConfigChange({ invertY: event.target.checked })}
              />
              <span>Inverse Y axis</span>
            </label>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function PlotPanel({
  title,
  subtitle,
  plotData,
  viewConfig,
  interactive,
  selectedOnly,
  pendingIds,
  confirmedSelections,
  onAddSelectionIds,
  onCancelPending,
  onConfirmPending,
  confirmButtonLabel = "Confirm",
  draftNote = "Each completed box or lasso action adds to the draft selection.",
  statusHint = null,
}) {
  const plotRef = useRef(null);
  const graphRef = useRef(null);
  const [dragMode, setDragMode] = useState("select");
  const [legendExpanded, setLegendExpanded] = useState(false);
  const colorInfo = useMemo(() => makeCategoryColors(plotData?.points?.color_values), [plotData]);

  useEffect(() => {
    if (!plotData || !plotRef.current) {
      return;
    }

    const graphDiv = plotRef.current;
    graphRef.current = graphDiv;

    const traces = buildPlotTraces({
      plotData,
      baseColors: colorInfo.colors,
      viewConfig,
      interactive,
      selectedOnly,
      pendingIds,
      confirmedSelections,
    });

    const layout = {
      dragmode: interactive ? dragMode : false,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.72)",
      margin: { l: 58, r: 20, t: 18, b: 52 },
      xaxis: {
        title: plotData.x_label,
        zeroline: false,
        gridcolor: "#dfe7f0",
        linecolor: "#c7d4e2",
        autorange: viewConfig.invertX ? "reversed" : true,
      },
      yaxis: {
        title: plotData.y_label,
        zeroline: false,
        gridcolor: "#dfe7f0",
        linecolor: "#c7d4e2",
        autorange: viewConfig.invertY ? "reversed" : true,
      },
      font: {
        family: "IBM Plex Sans, sans-serif",
        color: "#29415a",
      },
      showlegend: false,
    };

    const config = {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      scrollZoom: true,
    };

    Plotly.react(graphDiv, traces, layout, config);

    if (interactive) {
      graphDiv.removeAllListeners?.("plotly_selected");
      graphDiv.on("plotly_selected", (event) => {
        const ids = event?.points
          ? event.points
            .filter((point) => point.curveNumber === 0)
            .map((point) => Number(point.customdata ?? point.pointIndex))
          : [];
        if (ids.length) {
          onAddSelectionIds(ids);
        }
      });
    }
  }, [plotData, viewConfig, interactive, selectedOnly, pendingIds, confirmedSelections, colorInfo, dragMode, onAddSelectionIds]);

  useEffect(() => {
    return () => {
      if (graphRef.current) {
        Plotly.purge(graphRef.current);
      }
    };
  }, []);

  const legendItems = legendExpanded ? colorInfo.legend : colorInfo.legend.slice(0, 18);

  if (!plotData) {
    return (
      <section className="plot-shell panel-card empty-plot">
        <p>No embedding is displayed yet.</p>
        <span>Upload a dataset or choose a sample file from the right sidebar.</span>
      </section>
    );
  }

  return (
    <section className={classNames("plot-shell", "panel-card", !interactive && "plot-shell-readonly")}>
      <div className="plot-header">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <div className="toolbar-meta">
          <span>{plotData.embedding_key}</span>
          <span>{interactive ? `${pendingIds.length} pending` : `${confirmedSelections.length} confirmed`}</span>
        </div>
      </div>

      {interactive ? (
        <div className="selection-action-bar">
          <div className="plot-toolbar">
            <div className="toolbar-group">
              <button type="button" className={classNames("ghost-button", dragMode === "select" && "active")} onClick={() => setDragMode("select")}>
                Box Select
              </button>
              <button type="button" className={classNames("ghost-button", dragMode === "lasso" && "active")} onClick={() => setDragMode("lasso")}>
                Lasso Tool
              </button>
              <button type="button" className={classNames("ghost-button", dragMode === "pan" && "active")} onClick={() => setDragMode("pan")}>
                Pan
              </button>
            </div>
            <div className="toolbar-group">
              <button type="button" className="ghost-button" onClick={onCancelPending} disabled={!pendingIds.length}>
                Cancel
              </button>
              <button type="button" onClick={onConfirmPending} disabled={!pendingIds.length}>
                {confirmButtonLabel}
              </button>
            </div>
          </div>
          <p className="draft-note">{draftNote}</p>
        </div>
      ) : (
        <div className="readonly-note">
          <span>{statusHint || "Preview only"}</span>
          <span>{selectedOnly ? "Confirmed classes stay vivid while other cells turn grey." : "Confirmed classes overlay on top of the original coloring."}</span>
        </div>
      )}

      <div className="plot-stage">
        <div ref={plotRef} className="plot-area" />
      </div>

      <div className="legend-wrap">
        {legendItems.map((item) => (
          <div key={item.label} className="legend-item">
            <span className="legend-swatch" style={{ backgroundColor: item.color }} />
            <span>{item.label}</span>
          </div>
        ))}
        {colorInfo.legend.length > 18 ? (
          <button type="button" className="legend-toggle" onClick={() => setLegendExpanded((value) => !value)}>
            {legendExpanded ? "Collapse categories" : `+${colorInfo.legend.length - 18} more categories`}
          </button>
        ) : null}
      </div>
    </section>
  );
}

function AnalysisPanel({
  summary,
  activeSelection,
  confirmedSelections,
  analysisConfig,
  onConfigChange,
  onRunLassoView,
  onRunDownsample,
  onShowLassoViewCode,
  onShowDownsampleCode,
  onDownloadArtifact,
  job,
  disabled,
}) {
  const analysisBusy = disabled || (job && !["completed", "failed"].includes(job.status));
  const availableEmbeddings = summary?.available_embeddings || [];
  const selectedLassoViewSelection = confirmedSelections.find((selection) => selection.id === analysisConfig.lassoViewSelectionId) || activeSelection;

  return (
    <section className="panel-card analysis-panel">
      <div className="section-heading">
        <h2>Analysis</h2>
        <p>Run selection-aware analysis directly under the UMAP workspace. View 1 stays on the original dataset, while View 2 shows derived results.</p>
      </div>

      <div className="analysis-stack">
        <div className="analysis-block">
          <div className="analysis-header">
            <strong>Lasso-View</strong>
            <span>{selectedLassoViewSelection ? `${selectedLassoViewSelection.ids.length} selected cells` : "No target class"}</span>
          </div>
          <p>Pick which selection class to refine, expand it with the compiled LassoView backend, and also save the propagated set back into the selection list as `Refining n`.</p>
          <label>
            <span>Selection class</span>
            <select
              value={analysisConfig.lassoViewSelectionId}
              onChange={(event) => onConfigChange({ lassoViewSelectionId: event.target.value })}
              disabled={analysisBusy || !confirmedSelections.length}
            >
              <option value="">Select one class</option>
              {confirmedSelections.map((selection) => (
                <option key={selection.id} value={selection.id}>
                  {selection.displayName || selection.variableName}
                </option>
              ))}
            </select>
          </label>
          <label className="checkbox-line">
            <input
              type="checkbox"
              checked={analysisConfig.doCorrect}
              onChange={(event) => onConfigChange({ doCorrect: event.target.checked })}
              disabled={analysisBusy}
            />
            <span>Use corrected output</span>
          </label>
          <div className="toolbar-group">
            <button type="button" onClick={onRunLassoView} disabled={analysisBusy || !selectedLassoViewSelection}>
              Run Lasso-View
            </button>
            <button type="button" className="ghost-button" onClick={onShowLassoViewCode} disabled={!selectedLassoViewSelection}>
              Show Code
            </button>
          </div>
        </div>

        <div className="analysis-block">
          <div className="analysis-header">
            <strong>Downsample</strong>
            <span>{summary?.default_embedding || "No embedding"}</span>
          </div>
          <div className="analysis-grid">
            <label>
              <span>Sample rate</span>
              <input
                type="number"
                min="0.001"
                max="1"
                step="0.01"
                value={analysisConfig.sampleRate}
                onChange={(event) => onConfigChange({ sampleRate: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>
            <label>
              <span>Uniform share</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={analysisConfig.uniformRate}
                onChange={(event) => onConfigChange({ uniformRate: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>
            <label>
              <span>Leiden resolution</span>
              <input
                type="number"
                min="0.1"
                max="5"
                step="0.1"
                value={analysisConfig.leidenResolution}
                onChange={(event) => onConfigChange({ leidenResolution: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>
          </div>
          <div className="toolbar-group">
            <button type="button" onClick={onRunDownsample} disabled={analysisBusy || !availableEmbeddings.length}>
              Run Downsample
            </button>
            <button type="button" className="ghost-button" onClick={onShowDownsampleCode} disabled={!availableEmbeddings.length}>
              Show Code
            </button>
          </div>
        </div>

        <div className="analysis-status">
          <div className="analysis-header">
            <strong>Job status</strong>
            <span>{job ? job.status : "idle"}</span>
          </div>
          <p>{job?.message || "No analysis job has been started yet."}</p>
          {job ? <div className="progress-bar"><span style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} /></div> : null}
          {job?.status === "completed" ? (
            <div className="toolbar-group">
              <button type="button" className="ghost-button" onClick={() => onDownloadArtifact("result_h5ad")}>
                Download h5ad
              </button>
              {job?.result_info?.mapping_path ? (
                <button type="button" className="ghost-button" onClick={() => onDownloadArtifact("mapping")}>
                  Download mapping
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function App() {
  const [summary, setSummary] = useState(null);
  const [viewTwoSummary, setViewTwoSummary] = useState(null);
  const [viewTwoSource, setViewTwoSource] = useState({ analysisType: null, jobId: null, interactiveKind: null });
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Ready to explore a single-cell dataset.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedOnly, setSelectedOnly] = useState(true);
  const [openSection, setOpenSection] = useState("setting1");
  const [pendingIds, setPendingIds] = useState([]);
  const [viewTwoPendingIds, setViewTwoPendingIds] = useState([]);
  const [confirmedSelections, setConfirmedSelections] = useState([]);
  const [activeSelectionId, setActiveSelectionId] = useState(null);
  const [modalState, setModalState] = useState({ open: false, title: "", text: "" });
  const [copied, setCopied] = useState(false);
  const [analysisJob, setAnalysisJob] = useState(null);
  const [analysisConfig, setAnalysisConfig] = useState({
    sampleRate: 0.1,
    uniformRate: 0.5,
    leidenResolution: 1.0,
    doCorrect: true,
    lassoViewSelectionId: "",
    reconstructEmbeddingKey: "",
    encoderLayersKey: layersToKey([256, 64]),
    discriminatorLayersKey: layersToKey([256, 64]),
    pretrainEpoch: 20,
    trainingEpoch: 20,
    isPca: true,
    lambdaAttention: 0.1,
    lambdaRef: 0.3,
    nClusters: "",
  });

  const [viewOneConfig, setViewOneConfig] = useState(createViewConfig(null, null));
  const [viewTwoConfig, setViewTwoConfig] = useState(createViewConfig(null, null, { opacity: 0.92 }));
  const [viewOneData, setViewOneData] = useState(null);
  const [viewTwoData, setViewTwoData] = useState(null);

  const datasetId = summary?.dataset_id;
  const viewTwoDatasetId = viewTwoSummary?.dataset_id;
  const activeSelection = useMemo(
    () => confirmedSelections.find((selection) => selection.id === activeSelectionId) || confirmedSelections[confirmedSelections.length - 1] || null,
    [confirmedSelections, activeSelectionId],
  );
  const lassoViewSelection = useMemo(
    () => confirmedSelections.find((selection) => selection.id === analysisConfig.lassoViewSelectionId) || activeSelection,
    [confirmedSelections, analysisConfig.lassoViewSelectionId, activeSelection],
  );
  const analysisRunning = Boolean(analysisJob && !["completed", "failed"].includes(analysisJob.status));
  const viewTwoInteractive = viewTwoSource.interactiveKind === "downsample";
  const noDefaultEmbedding = summary?.needs_umap_choice && !viewOneData;

  const duplicateStats = useMemo(() => {
    const occurrenceMap = new Map();
    confirmedSelections.forEach((selection) => {
      selection.ids.forEach((id) => {
        occurrenceMap.set(id, (occurrenceMap.get(id) || 0) + 1);
      });
    });

    let repeatedDistinct = 0;
    let repeatedAssignments = 0;
    occurrenceMap.forEach((count) => {
      if (count > 1) {
        repeatedDistinct += 1;
        repeatedAssignments += count - 1;
      }
    });

    const overlapCountMap = new Map();
    confirmedSelections.forEach((selection) => {
      let overlapCount = 0;
      selection.ids.forEach((id) => {
        if ((occurrenceMap.get(id) || 0) > 1) {
          overlapCount += 1;
        }
      });
      overlapCountMap.set(selection.id, overlapCount);
    });

    return {
      repeatedDistinct,
      repeatedAssignments,
      overlapCountMap,
    };
  }, [confirmedSelections]);

  const ingestResponse = (payload) => {
    const nextViewOneConfig = createViewConfig(payload.summary, payload.plot);
    const nextViewTwoConfig = createViewConfig(payload.summary, payload.plot, { opacity: 0.92 });

    setSummary(payload.summary);
    setViewTwoSummary(payload.summary);
    setViewTwoSource({ analysisType: null, jobId: null, interactiveKind: null });
    setPendingIds([]);
    setViewTwoPendingIds([]);
    setConfirmedSelections([]);
    setActiveSelectionId(null);
    setAnalysisJob(null);
    setAnalysisConfig((current) => ({
      ...current,
      reconstructEmbeddingKey: payload.summary?.default_embedding || payload.summary?.available_embeddings?.[0] || "",
    }));
    setViewOneConfig(nextViewOneConfig);
    setViewTwoConfig(nextViewTwoConfig);
    setViewOneData(payload.plot);
    setViewTwoData(payload.plot);
  };

  const fetchPlot = async (targetDatasetId, config) => {
    const response = await fetch(`/api/datasets/${targetDatasetId}/plot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        embedding_key: config.embeddingKey || null,
        color_by: config.colorBy || null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to update plot.");
    }
    return payload;
  };

  const fetchJob = async (jobId) => {
    const response = await fetch(`/api/analysis-jobs/${jobId}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to fetch analysis job.");
    }
    return payload;
  };

  const applyCompletedJob = (jobSnapshot) => {
    if (!jobSnapshot.result_summary || !jobSnapshot.result_plot) {
      return;
    }
    const interactiveKind = jobSnapshot.interactive_kind
      || jobSnapshot.result_info?.interactive_kind
      || (jobSnapshot.analysis_type === "downsample" ? "downsample" : null);
    const nextViewTwoConfig = createViewConfig(jobSnapshot.result_summary, jobSnapshot.result_plot, { opacity: 0.92 });
    setViewTwoSummary(jobSnapshot.result_summary);
    setViewTwoData(jobSnapshot.result_plot);
    setViewTwoConfig(nextViewTwoConfig);
    setViewTwoPendingIds([]);
    setViewTwoSource({
      analysisType: jobSnapshot.analysis_type,
      jobId: jobSnapshot.job_id,
      interactiveKind,
    });
  };

  useEffect(() => {
    if (!analysisJob?.job_id || ["completed", "failed"].includes(analysisJob.status)) {
      return undefined;
    }

    let cancelled = false;
    const intervalId = window.setInterval(async () => {
      try {
        const snapshot = await fetchJob(analysisJob.job_id);
        if (cancelled) {
          return;
        }
        setAnalysisJob(snapshot);
        if (snapshot.status === "completed") {
          applyCompletedJob(snapshot);
          if (snapshot.analysis_type === "lasso_view" && Array.isArray(snapshot.result_info?.expanded_ids)) {
            commitConfirmedSelection(
              snapshot.result_info.expanded_ids,
              (selection) => `${selection.displayName} was added with ${selection.ids.length.toLocaleString()} propagated cells.`,
              {
                kind: "refining",
                displayLabelPrefix: "Refining",
                variablePrefix: "refine_list",
              },
            );
            return;
          }
          setStatus(`${snapshot.analysis_type} completed and loaded into View 2.`);
        } else if (snapshot.status === "failed") {
          setError(snapshot.error || snapshot.message || "Analysis failed.");
          setStatus("Analysis failed.");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      }
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [analysisJob?.job_id, analysisJob?.status]);

  const loadSample = async (sampleName) => {
    setBusy(true);
    setError("");
    setStatus(`Loading ${sampleName}...`);
    try {
      const response = await fetch(`/api/load-sample?name=${encodeURIComponent(sampleName)}`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load sample dataset.");
      }
      ingestResponse(payload);
      setStatus(`Loaded ${payload.summary.dataset_name} with ${payload.summary.n_obs.toLocaleString()} cells.`);
    } catch (err) {
      setError(err.message);
      setStatus("Sample loading failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose an h5ad file first.");
      return;
    }

    setBusy(true);
    setError("");
    setStatus(`Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Upload failed.");
      }
      ingestResponse(payload);
      setStatus(`Loaded ${payload.summary.dataset_name} with ${payload.summary.n_obs.toLocaleString()} cells.`);
    } catch (err) {
      setError(err.message);
      setStatus("Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleViewConfigChange = async (viewName, patch) => {
    const currentConfig = viewName === "setting1" ? viewOneConfig : viewTwoConfig;
    const nextConfig = { ...currentConfig, ...patch };
    const setConfig = viewName === "setting1" ? setViewOneConfig : setViewTwoConfig;
    const setPlot = viewName === "setting1" ? setViewOneData : setViewTwoData;
    const targetDatasetId = viewName === "setting1" ? datasetId : viewTwoDatasetId;

    setConfig(nextConfig);

    if (!targetDatasetId) {
      return;
    }
    if (!("embeddingKey" in patch) && !("colorBy" in patch)) {
      return;
    }

    setBusy(true);
    setError("");
    try {
      const payload = await fetchPlot(targetDatasetId, nextConfig);
      setPlot(payload);
      setStatus(`Updated ${viewName} to ${payload.embedding_key}${payload.color_by ? ` colored by ${payload.color_by}` : ""}.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleComputeUmap = async () => {
    if (!datasetId) {
      return;
    }
    setBusy(true);
    setError("");
    setStatus("Computing UMAP with Scanpy...");
    try {
      const response = await fetch(`/api/datasets/${datasetId}/compute-umap`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "UMAP computation failed.");
      }
      ingestResponse(payload);
      setStatus("UMAP computed and displayed.");
    } catch (err) {
      setError(err.message);
      setStatus("UMAP computation failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleAddSelectionIds = (ids) => {
    setPendingIds((current) => {
      const next = arrayUnion(current, ids);
      setStatus(`Draft selection now contains ${next.length.toLocaleString()} cells. Continue selecting, then Confirm or Cancel.`);
      return next;
    });
  };

  const handleAddViewTwoSelectionIds = (ids) => {
    setViewTwoPendingIds((current) => {
      const next = arrayUnion(current, ids);
      setStatus(`Downsample draft now contains ${next.length.toLocaleString()} sampled cells. Continue selecting or map them back to the original dataset.`);
      return next;
    });
  };

  const handleCancelPending = () => {
    setPendingIds([]);
    setStatus("Draft selection cleared.");
  };

  const commitConfirmedSelection = (ids, messageBuilder, options = {}) => {
    if (!ids.length) {
      return;
    }
    setConfirmedSelections((current) => {
      const kind = options.kind || "selection";
      const displayLabelPrefix = options.displayLabelPrefix || "Selection";
      const variablePrefix = options.variablePrefix || "select_list";
      const kindIndex = current.filter((selection) => selection.kind === kind).length;
      const displayName = options.displayName || `${displayLabelPrefix} ${kindIndex + 1}`;
      const nextSelection = {
        id: `selection-${Date.now()}-${current.length}`,
        kind,
        displayName,
        variableName: options.variableName || `${variablePrefix}${kindIndex + 1}`,
        ids: ids.slice(),
        color: SELECTION_COLORS[current.length % SELECTION_COLORS.length],
      };
      const next = [...current, nextSelection];
      setActiveSelectionId(nextSelection.id);
      setStatus(messageBuilder(nextSelection));
      return next;
    });
  };

  const handleConfirmPending = () => {
    if (!pendingIds.length) {
      return;
    }
    const nextIds = pendingIds.slice();
    commitConfirmedSelection(nextIds, (selection) => `Confirmed ${selection.variableName} with ${selection.ids.length.toLocaleString()} cells.`, {
      kind: "selection",
      displayLabelPrefix: "Selection",
      variablePrefix: "select_list",
    });
    setPendingIds([]);
  };

  const openTextModal = (title, text) => {
    setCopied(false);
    setModalState({ open: true, title, text });
  };

  const closeTextModal = () => {
    setModalState({ open: false, title: "", text: "" });
  };

  const copyModalContent = async () => {
    try {
      await navigator.clipboard.writeText(modalState.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      setError("Failed to copy selection content to the clipboard.");
    }
  };

  const formatSelectionLine = (selection, index) => `${selection.variableName || `select_list${index + 1}`}=[${selection.ids.join(",")}]`;

  const handleShowAllIds = () => {
    if (!confirmedSelections.length) {
      return;
    }
    openTextModal(
      "All selection classes",
      confirmedSelections.map((selection, index) => formatSelectionLine(selection, index)).join("\n"),
    );
  };

  const handleShowSingleSelection = (selection) => {
    const index = confirmedSelections.findIndex((item) => item.id === selection.id);
    openTextModal(`IDs for ${selection.variableName}`, formatSelectionLine(selection, index));
  };

  const handleDeleteSelection = (selectionId) => {
    setConfirmedSelections((current) => {
      const next = normalizeSelectionCatalog(current.filter((selection) => selection.id !== selectionId));
      setActiveSelectionId(next.length ? next[Math.max(0, next.length - 1)].id : null);
      return next;
    });
    setStatus("One confirmed selection class was deleted.");
  };

  const handleDeleteAllSelections = () => {
    setPendingIds([]);
    setViewTwoPendingIds([]);
    setConfirmedSelections([]);
    setActiveSelectionId(null);
    setStatus("All selection classes were removed.");
  };

  const exportSelectionJsonLines = () => {
    if (!confirmedSelections.length) {
      return;
    }
    const content = confirmedSelections.map((selection) => JSON.stringify(selection.ids)).join("\n");
    const blob = new Blob([content], { type: "application/x-ndjson" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "selection_lists.jsonl";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const submitAnalysisJob = async (payload, pendingMessage) => {
    if (!datasetId) {
      return;
    }
    setBusy(true);
    setError("");
    setStatus(pendingMessage);
    try {
      const response = await fetch(`/api/datasets/${datasetId}/analysis-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const jobSnapshot = await response.json();
      if (!response.ok) {
        throw new Error(jobSnapshot.detail || "Failed to create analysis job.");
      }
      setAnalysisJob(jobSnapshot);
      setViewTwoPendingIds([]);
      setViewTwoSource({
        analysisType: jobSnapshot.analysis_type,
        jobId: jobSnapshot.job_id,
        interactiveKind: null,
      });
      setStatus(jobSnapshot.message || "Analysis job queued.");

      // If the job already completed synchronously, apply it immediately
      if (jobSnapshot.status === "completed") {
        applyCompletedJob(jobSnapshot);
      }
    } catch (err) {
      setError(err.message);
      setStatus("Analysis job failed to start.");
    } finally {
      setBusy(false);
    }
  };

  const handleRunLassoView = async () => {
    if (!lassoViewSelection) {
      setError("Choose one selection class for Lasso-View first.");
      return;
    }
    await submitAnalysisJob(
      {
        analysis_type: "lasso_view",
        selected_ids: lassoViewSelection.ids,
        embedding_key: viewOneConfig.embeddingKey || summary?.default_embedding || null,
        color_by: viewOneConfig.colorBy || summary?.default_color_by || null,
        obs_col: viewOneConfig.colorBy || summary?.default_color_by || null,
        leiden_resolution: analysisConfig.leidenResolution,
        do_correct: analysisConfig.doCorrect,
      },
      "Submitting Lasso-View job...",
    );
  };

  const handleRunDownsample = async () => {
    await submitAnalysisJob(
      {
        analysis_type: "downsample",
        embedding_key: viewOneConfig.embeddingKey || summary?.default_embedding || null,
        color_by: viewOneConfig.colorBy || summary?.default_color_by || null,
        sample_rate: analysisConfig.sampleRate,
        uniform_rate: analysisConfig.uniformRate,
        leiden_resolution: analysisConfig.leidenResolution,
      },
      "Submitting downsample job...",
    );
  };

  const handleShowLassoViewCode = () => openTextModal("Lasso-View code", buildLassoViewCode({ activeSelection: lassoViewSelection, summary, viewOneConfig, analysisConfig }));
  const handleShowDownsampleCode = () => openTextModal("Downsample code", buildDownsampleCode({ summary, viewOneConfig, analysisConfig }));

  const handleRecoverFromDownsample = async () => {
    if (!viewTwoSource.jobId || !viewTwoPendingIds.length) {
      return;
    }
    setBusy(true);
    setError("");
    setStatus("Recovering selected downsampled cells back to the original dataset...");
    try {
      const response = await fetch(`/api/analysis-jobs/${viewTwoSource.jobId}/recover-selection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: viewTwoPendingIds }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to recover selection.");
      }
      const recoveredIds = payload.ids || [];
      commitConfirmedSelection(
        recoveredIds,
        (selection) => `Recovered ${selection.ids.length.toLocaleString()} original cells from the downsampled selection into ${selection.variableName}.`,
      );
      setViewTwoPendingIds([]);
    } catch (err) {
      setError(err.message);
      setStatus("Recovering downsampled selection failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleDownloadArtifact = (artifact) => {
    if (!analysisJob?.job_id) {
      return;
    }
    window.open(`/api/analysis-jobs/${analysisJob.job_id}/download/${artifact}`, "_blank");
  };

  const statusRows = [
    { label: "Dataset", value: summary?.dataset_name || "None loaded" },
    { label: "Cells", value: summary ? summary.n_obs.toLocaleString() : "0" },
    { label: "Genes", value: summary ? summary.n_vars.toLocaleString() : "0" },
    { label: "View 1", value: viewOneData ? `${viewOneData.embedding_key} | ${viewOneData.color_by || "No color"}` : "No plot" },
    { label: "View 2", value: viewTwoData ? `${viewTwoData.embedding_key} | ${viewTwoData.color_by || "No color"}` : "No plot" },
    { label: "View 2 dataset", value: viewTwoSummary?.dataset_name || "Original dataset" },
    { label: "Analysis job", value: analysisJob ? `${analysisJob.analysis_type} | ${analysisJob.status}` : "Idle" },
    { label: "Draft selection", value: `${pendingIds.length.toLocaleString()} cells pending` },
    { label: "Confirmed classes", value: `${confirmedSelections.length.toLocaleString()}` },
    {
      label: "Repeated selections",
      value: duplicateStats.repeatedDistinct
        ? `${duplicateStats.repeatedDistinct.toLocaleString()} repeated cells, ${duplicateStats.repeatedAssignments.toLocaleString()} extra assignments`
        : "No repeated cells across confirmed classes",
    },
  ];

  return (
    <main className="app-shell">
      <section className="workspace-shell">
        <section className="viewer-column">
          <div className="board-head panel-card">
            <div>
              <p className="eyebrow">Single-Cell Analysis and Visualization Platform</p>
              <h1>{summary?.dataset_name || "Scav-P"}</h1>
              <p className="hero-copy">
                View 1 stays anchored to the original dataset. Analysis jobs create derived results that load into View 2, where downsampled outputs can be mapped back into new original-dataset selections.
              </p>
            </div>
            <div className="hero-badges">
              <span>Background jobs</span>
              <span>env1 + rsc_2502</span>
              <span>Selection classes</span>
            </div>
          </div>

          {noDefaultEmbedding ? (
            <section className="choice-banner panel-card">
              <div>
                <h3>No UMAP found</h3>
                <p>This dataset does not contain a usable UMAP embedding yet. You can compute UMAP with Scanpy or choose another two-dimensional entry in Setting 1.</p>
              </div>
              <button type="button" onClick={handleComputeUmap} disabled={busy || analysisRunning}>Compute UMAP</button>
            </section>
          ) : null}

          <section className="plot-grid">
            <PlotPanel
              title="View 1"
              subtitle="Interactive selection canvas on the original dataset"
              plotData={viewOneData}
              viewConfig={viewOneConfig}
              interactive={true}
              selectedOnly={false}
              pendingIds={pendingIds}
              confirmedSelections={confirmedSelections}
              onAddSelectionIds={handleAddSelectionIds}
              onCancelPending={handleCancelPending}
              onConfirmPending={handleConfirmPending}
              confirmButtonLabel="Confirm"
              draftNote="Each completed box or lasso action adds to the draft selection. Use Confirm to create a class, or Cancel to clear the draft."
            />
            <PlotPanel
              title="View 2"
              subtitle={viewTwoSource.analysisType ? `Derived result: ${viewTwoSource.analysisType}` : "Read-only comparison preview"}
              plotData={viewTwoData}
              viewConfig={viewTwoConfig}
              interactive={viewTwoInteractive}
              selectedOnly={viewTwoInteractive ? false : selectedOnly}
              pendingIds={viewTwoPendingIds}
              confirmedSelections={viewTwoInteractive ? [] : confirmedSelections}
              onAddSelectionIds={viewTwoInteractive ? handleAddViewTwoSelectionIds : () => {}}
              onCancelPending={viewTwoInteractive ? () => setViewTwoPendingIds([]) : () => {}}
              onConfirmPending={viewTwoInteractive ? handleRecoverFromDownsample : () => {}}
              confirmButtonLabel={viewTwoInteractive ? "Map To Original Dataset" : "Confirm"}
              draftNote={viewTwoInteractive ? "Select sampled cells here, then map them back into a new confirmed selection class on the original dataset." : "Preview only."}
              statusHint={viewTwoSource.analysisType ? `Result from ${viewTwoSource.analysisType}` : "Preview only"}
            />
          </section>

          <SelectionTabs
            selections={confirmedSelections}
            activeSelectionId={activeSelectionId}
            overlapCountMap={duplicateStats.overlapCountMap}
            onSelect={setActiveSelectionId}
            onShowIds={handleShowSingleSelection}
            onDelete={handleDeleteSelection}
          />

          <AnalysisPanel
            summary={summary}
            activeSelection={activeSelection}
            confirmedSelections={confirmedSelections}
            analysisConfig={analysisConfig}
            onConfigChange={(patch) => setAnalysisConfig((current) => ({ ...current, ...patch }))}
            onRunLassoView={handleRunLassoView}
            onRunDownsample={handleRunDownsample}
            onShowLassoViewCode={handleShowLassoViewCode}
            onShowDownsampleCode={handleShowDownsampleCode}
            onDownloadArtifact={handleDownloadArtifact}
            job={analysisJob}
            disabled={busy}
          />
        </section>

        <aside className="sidebar-column">
          <form className="side-card" onSubmit={handleUpload}>
            <div className="section-heading">
              <h2>Upload</h2>
              <p>Bring your own h5ad file or start from a local sample.</p>
            </div>
            <label className="file-drop">
              <input type="file" accept=".h5ad" onChange={(event) => setFile(event.target.files?.[0] || null)} />
              <span>{file ? file.name : "Choose a local .h5ad file"}</span>
            </label>
            <div className="button-stack">
              <button type="submit" disabled={busy || analysisRunning}>Upload h5ad file</button>
              <button type="button" className="ghost-button" onClick={() => loadSample("sc_sampled.h5ad")} disabled={busy || analysisRunning}>Load sample</button>
              <button type="button" className="ghost-button" onClick={() => loadSample("sc_sample_large.h5ad")} disabled={busy || analysisRunning}>Load large sample</button>
            </div>
            <p className="status-line">{status}</p>
            {error ? <p className="error-line">{error}</p> : null}
          </form>

          <section className="side-card">
            <div className="section-heading">
              <h2>Current status</h2>
              <p>Each row wraps naturally so long metadata stays readable.</p>
            </div>
            <div className="status-list">
              {statusRows.map((row) => (
                <div key={row.label} className="status-row">
                  <span className="status-key">{row.label}</span>
                  <strong className="status-value">{row.value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className={classNames("side-card", "preview-mode-card", selectedOnly && "preview-mode-active")}>
            <div className="section-heading">
              <h2>Preview mode</h2>
              <p>This affects View 2 only when it is showing a non-interactive result.</p>
            </div>
            <button
              type="button"
              className={classNames("mode-button", selectedOnly && "mode-button-active")}
              onClick={() => setSelectedOnly((value) => !value)}
              disabled={viewTwoInteractive}
            >
              {selectedOnly ? "Selected only: ON" : "Selected only: OFF"}
            </button>
            <p className="mode-caption">
              When enabled, unselected cells turn grey in View 2. Confirmed classes still keep their dedicated colors.
            </p>
          </section>

          <section className="side-card">
            <div className="section-heading">
              <h2>Selection tools</h2>
              <p>Work with all confirmed selection classes at once.</p>
            </div>
            <div className="selection-summary">
              <strong>{confirmedSelections.length}</strong>
              <span>confirmed classes</span>
            </div>
            <div className="button-stack">
              <button type="button" className="ghost-button" onClick={handleShowAllIds} disabled={!confirmedSelections.length}>Show all IDs</button>
              <button type="button" className="ghost-button" onClick={exportSelectionJsonLines} disabled={!confirmedSelections.length}>Download JSON lines</button>
              <button type="button" className="ghost-button danger-button" onClick={handleDeleteAllSelections} disabled={!confirmedSelections.length && !pendingIds.length && !viewTwoPendingIds.length}>Delete all selections</button>
            </div>
          </section>

          <SettingsPanel
            title="Setting 1"
            open={openSection === "setting1"}
            onToggle={() => setOpenSection((value) => (value === "setting1" ? "" : "setting1"))}
            summary={summary}
            config={viewOneConfig}
            onConfigChange={(patch) => handleViewConfigChange("setting1", patch)}
            busy={busy || analysisRunning}
          />

          <SettingsPanel
            title="Setting 2"
            open={openSection === "setting2"}
            onToggle={() => setOpenSection((value) => (value === "setting2" ? "" : "setting2"))}
            summary={viewTwoSummary}
            config={viewTwoConfig}
            onConfigChange={(patch) => handleViewConfigChange("setting2", patch)}
            busy={busy}
          />
        </aside>
      </section>

      <SelectionTextModal
        open={modalState.open}
        title={modalState.title}
        text={modalState.text}
        copied={copied}
        onClose={closeTextModal}
        onCopy={copyModalContent}
      />
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
