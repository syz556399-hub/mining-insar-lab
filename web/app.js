const $ = id => document.getElementById(id);
let request, defaults, current, activeFace = 0,
  revision = 0,
  timer, jobId;
const clone = x => JSON.parse(JSON.stringify(x));

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  setTimeout(() => $('toast').hidden = true, 4500);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  if (!response.ok) throw Error((await response.json()).error || '请求失败');
  return response;
}
const groups = {
  'time-fields': [
    ['day_before', '前期 / 天'],
    ['day_after', '后期 / 天']
  ],
  'face-fields': [
    ['east_m', '中心东向 / m'],
    ['north_m', '中心北向 / m'],
    ['length_m', '推进长度 / m'],
    ['width_m', '工作面宽度 / m'],
    ['bearing_deg', '推进角度 / °'],
    ['depth_m', '埋深 / m'],
    ['advance_m_day', '推进速度 / m/天'],
    ['start_day', '开采开始 / 天']
  ],
  'response-fields': [
    ['thickness_m', '煤层厚度 / m'],
    ['subsidence_factor', '下沉系数'],
    ['influence_tangent', '影响角正切'],
    ['response_days', '响应时间 / 天']
  ],
  'environment-fields': [
    ['relief_m', '地形起伏 / m'],
    ['river_width_m', '河流宽度 / m'],
    ['dem_error_m', 'DEM 误差 / m']
  ],
  'observation-fields': [
    ['land_coherence', '陆地初始相干性'],
    ['water_coherence', '水体相干性'],
    ['decorrelation_days', '失相干时间 / 天（0关闭）'],
    ['slope_coherence_loss', '坡度影响系数'],
    ['looks', '独立观测次数'],
    ['valid_coherence_threshold', '训练有效区相干阈值']
  ],
  'signal-fields': [
    ['wavelength_m', '波长 / mm', 1000],
    ['incidence_deg', '入射角 / °'],
    ['radar_azimuth_deg', '卫星方位 / °'],
    ['horizontal_factor', '水平移动系数'],
    ['atmosphere_mm', '大气幅度 / mm'],
    ['atmosphere_scale_m', '大气尺度 / m'],
    ['orbit_cycles', '东西坡度 / 周'],
    ['baseline_m', '垂直基线 / m'],
    ['slant_range_m', '斜距 / m'],
    ['height_atmosphere_mm', '分层大气 / mm每100m']
  ],
  'general-fields': [
    ['extent_m', '场景边长 / m'],
    ['target_threshold_mm', '标签阈值 / mm'],
    ['segments', '推进分段数'],
    ['seed', '随机种子']
  ]
};
const simpleFields = new Set(['day_before', 'day_after', 'length_m', 'width_m', 'depth_m', 'advance_m_day', 'relief_m', 'river_width_m', 'land_coherence', 'extent_m', 'target_threshold_mm', 'seed']);

function setMode(mode) {
  const advanced = mode === 'advanced';
  document.body.classList.toggle('simple-mode', !advanced);
  $('simple-mode').setAttribute('aria-pressed', String(!advanced));
  $('advanced-mode').setAttribute('aria-pressed', String(advanced));
  $('mode-hint').textContent = advanced ? '全部参数 · 按分组展开精细调整' : '常用参数 · 未显示的参数仍按当前值计算';
  try {
    localStorage.setItem('mining-parameter-mode', advanced ? 'advanced' : 'simple');
  } catch (_) {}
  if (current && !$('scene-view').hidden) {
    drawPlan();
    drawProfile();
  }
}
$('simple-mode').onclick = () => setMode('simple');
$('advanced-mode').onclick = () => setMode('advanced');
let savedMode = 'simple';
try {
  savedMode = localStorage.getItem('mining-parameter-mode') || 'simple';
} catch (_) {}
setMode(savedMode);

function fields() {
  for (const [group, specs] of Object.entries(groups)) {
    const face = ['face-fields', 'response-fields'].includes(group);
    $(group).replaceChildren();
    for (const [key, label, mult = 1] of specs) {
      const wrapper = document.createElement('label');
      wrapper.className = 'field' + (simpleFields.has(key) ? '' : ' advanced-only');
      const span = document.createElement('span');
      span.textContent = label;
      const input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.id = key;
      input.value = (face ? request.faces[activeFace] : request.settings)[key] * mult;
      input.addEventListener('input', () => {
        if (input.value === '') return;
        (face ? request.faces[activeFace] : request.settings)[key] = Number(input.value) / mult;
        schedule();
      });
      wrapper.append(span, input);
      $(group).append(wrapper);
    }
  }
  $('terrain-enabled').checked = request.settings.terrain_enabled;
  $('water-enabled').checked = request.settings.water_enabled;
  $('topography-mode').value = request.settings.raw_topography ? 'raw' : 'differential';
  $('spatial-window').value = request.settings.spatial_window;
  $('face-enabled').checked = request.faces[activeFace].enabled;
  $('disturbed-patch').checked = request.settings.disturbed_patch;
  $('size').value = request.settings.size;
  $('face-a').classList.toggle('active', activeFace === 0);
  $('face-b').classList.toggle('active', activeFace === 1);
}

function schedule() {
  revision++;
  clearTimeout(timer);
  $('export').disabled = true;
  $('status').textContent = '等待更新…';
  timer = setTimeout(refresh, 350);
}
async function refresh() {
  const version = revision,
    snapshot = clone(request);
  $('status').textContent = '计算中…';
  try {
    const result = await (await api('/api/preview', snapshot)).json();
    if (version !== revision) return;
    current = result;
    render();
    $('status').textContent = '计算完成';
    $('export').disabled = false;
  } catch (error) {
    if (version === revision) {
      $('status').textContent = '参数需要调整';
      toast(error.message);
    }
  }
}

function render() {
  const m = current.metadata,
    s = m.request.settings;
  $('phase-image').src = current.images.observed;
  $('clean-image').src = current.images[$('phase-view').value];
  $('dem-image').src = current.images.dem;
  $('water-image').src = current.images.water;
  $('coherence-image').src = current.images.coherence;
  $('dem-range').textContent = `${m.dem_min_m.toFixed(0)}–${m.dem_max_m.toFixed(0)} m`;
  $('water-summary').textContent = `水体 ${m.water_percent.toFixed(1)}%`;
  $('coherence-title').textContent = s.complex_observation ? '估计相干性' : '圆周合向量长度';
  $('coherence-summary').textContent = s.complex_observation ? `模型均值 ${m.mean_model_coherence.toFixed(2)} · 估计均值 ${m.mean_estimated_coherence.toFixed(2)} · 有效区 ${m.valid_percent.toFixed(1)}%` : '兼容圆周噪声模式：此图不是 SAR 样本相干性。';
  $('displacement-image').src = current.images.displacement;
  $('mask-image').src = current.images[$('overlay').checked ? 'overlay' : 'mask'];
  $('peak').textContent = m.peak_delta_down_mm.toFixed(1) + ' mm';
  $('cycles').textContent = m.phase_cycles.toFixed(2) + ' 周';
  $('area').textContent = m.mask_percent.toFixed(1) + '%';
  $('overlap').textContent = m.overlap_pixels + ' px';
  $('heat-max').textContent = m.peak_delta_down_mm.toFixed(1) + ' mm';
  $('description').textContent = `第 ${s.day_before} → ${s.day_after} 天 · ${(s.extent_m/1000).toFixed(2)} km 场景 · ${s.size} × ${s.size} 像素`;
  $('mask-description').textContent = `垂直沉降增量 ≥ ${s.target_threshold_mm} mm 的区域为目标。`;
  $('profile-row').textContent = `第 ${m.profile_row} 行（从 0 计）`;
  $('warnings').hidden = !m.warnings.length;
  $('warnings').textContent = m.warnings.join(' ');
  drawPlan();
  drawProfile();
}

function canvas(id) {
  const c = $(id),
    r = c.getBoundingClientRect(),
    d = window.devicePixelRatio || 1;
  c.width = r.width * d;
  c.height = r.height * d;
  const ctx = c.getContext('2d');
  ctx.scale(d, d);
  return [ctx, r.width, r.height];
}

function drawPlan() {
  if (!current) return;
  const [ctx, w, h] = canvas('plan'), r = current.metadata.request, s = r.settings, scale = w / s.extent_m;
  ctx.fillStyle = '#eef4f3';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#dbe5e3';
  for (let i = 1; i < 8; i++) {
    ctx.beginPath();
    ctx.moveTo(w * i / 8, 0);
    ctx.lineTo(w * i / 8, h);
    ctx.moveTo(0, h * i / 8);
    ctx.lineTo(w, h * i / 8);
    ctx.stroke();
  }
  r.faces.forEach((f, i) => {
    if (!f.enabled) return;
    ctx.save();
    ctx.translate(w / 2 + f.east_m * scale, h / 2 - f.north_m * scale);
    ctx.rotate(-f.bearing_deg * Math.PI / 180);
    const l = f.length_m * scale,
      b = f.width_m * scale,
      progress = Math.min(1, Math.max(0, (s.day_after - f.start_day) * f.advance_m_day / f.length_m));
    ctx.fillStyle = i ? '#d3853d66' : '#087f8466';
    ctx.fillRect(-l / 2, -b / 2, l * progress, b);
    ctx.strokeStyle = i ? '#b66b25' : '#087f84';
    ctx.lineWidth = i === activeFace ? 3 : 1.5;
    ctx.strokeRect(-l / 2, -b / 2, l, b);
    ctx.fillStyle = '#243e43';
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(i ? 'B →' : 'A →', -12, 4);
    ctx.restore();
  });
  ctx.fillStyle = '#486263';
  ctx.font = '12px sans-serif';
  ctx.fillText('北 ↑', 12, 20);
  ctx.fillText(`${s.extent_m} m`, 12, h - 12);
}

function drawProfile() {
  if (!current) return;
  const [ctx, w, h] = canvas('profile'), left = 42, right = 12, top = 15, bottom = 27, p = current.profile;
  const max = Math.max(1, ...p.subsidence_after_m),
    extent = current.metadata.request.settings.extent_m;
  ctx.font = '11px sans-serif';
  ctx.fillStyle = '#617579';
  ctx.strokeStyle = '#e4eceb';
  for (let i = 0; i <= 4; i++) {
    let y = top + (h - top - bottom) * i / 4;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(w - right, y);
    ctx.stroke();
    ctx.fillText((max * (1 - i / 4)).toFixed(0), 2, y + 4);
  }
  [
    ['subsidence_before_m', '#92a3b0'],
    ['subsidence_after_m', '#087f84'],
    ['delta_down_m', '#d3853d']
  ].forEach(([key, color]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    p[key].forEach((v, i) => {
      const x = left + i / (p[key].length - 1) * (w - left - right),
        y = top + (1 - v / max) * (h - top - bottom);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillText((-extent / 2).toFixed(0), left, h - 6);
  ctx.fillText('0', (w + left) / 2, h - 6);
  ctx.fillText((extent / 2).toFixed(0), w - 35, h - 6);
}

function download(blob, name) {
  const url = URL.createObjectURL(blob),
    a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 15000);
}
$('phase-view').onchange = () => current && render();
$('overlay').onchange = () => current && render();
$('face-a').onclick = () => {
  activeFace = 0;
  fields();
  drawPlan();
};
$('face-b').onclick = () => {
  activeFace = 1;
  fields();
  drawPlan();
};
$('face-enabled').onchange = e => {
  request.faces[activeFace].enabled = e.target.checked;
  schedule();
};
$('disturbed-patch').onchange = e => {
  request.settings.disturbed_patch = e.target.checked;
  schedule();
};
$('terrain-enabled').onchange = e => {
  request.settings.terrain_enabled = e.target.checked;
  schedule();
};
$('water-enabled').onchange = e => {
  request.settings.water_enabled = e.target.checked;
  schedule();
};
$('topography-mode').onchange = e => {
  request.settings.raw_topography = e.target.value === 'raw';
  schedule();
};
$('spatial-window').onchange = e => {
  request.settings.spatial_window = Number(e.target.value);
  schedule();
};
$('size').onchange = e => {
  request.settings.size = Number(e.target.value);
  schedule();
};
$('no-noise').onclick = () => {
  Object.assign(request.settings, {
    atmosphere_mm: 0,
    orbit_cycles: 0,
    phase_spread_rad: 0,
    disturbed_patch: false,
    land_coherence: 1,
    water_coherence: 1,
    water_enabled: false,
    terrain_enabled: false,
    dem_error_m: 0,
    height_atmosphere_mm: 0,
    decorrelation_days: 0,
    slope_coherence_loss: 0,
    spatial_window: 1
  });
  fields();
  schedule();
};

function preset(name) {
  request = clone(defaults);
  if (name === 'single') request.faces[1].enabled = false;
  if (name === 'negative') request.faces.forEach(f => f.enabled = false);
  if (name === 'staggered') request.faces[1].start_day = 40;
  document.querySelectorAll('[data-preset]').forEach(b => b.classList.toggle('selected', b.dataset.preset === name));
  fields();
  schedule();
}
$('reset').onclick = () => preset('overlap');
document.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => preset(b.dataset.preset));
$('toggle-settings').onclick = () => {
  const expanded = document.querySelector('aside').classList.toggle('expanded');
  $('toggle-settings').textContent = expanded ? '收起设置' : '展开设置';
  $('toggle-settings').setAttribute('aria-expanded', expanded);
};
$('plan').onclick = e => {
  if (!current) return;
  const rect = $('plan').getBoundingClientRect(),
    extent = current.metadata.request.settings.extent_m;
  request.faces[activeFace].east_m = Math.round((e.clientX - rect.left) / rect.width * extent - extent / 2);
  request.faces[activeFace].north_m = Math.round(extent / 2 - (e.clientY - rect.top) / rect.height * extent);
  fields();
  schedule();
};
$('save-parameters').onclick = () => download(new Blob([JSON.stringify(request, null, 2)], {
  type: 'application/json'
}), 'mine_parameters.json');
$('load-parameters').onclick = () => $('parameter-file').click();
$('parameter-file').onchange = async e => {
  try {
    const file = e.target.files[0];
    if (!file) return;
    const proposed = JSON.parse(await file.text());
    const checked = await (await api('/api/preview', proposed)).json();
    request = checked.metadata.request;
    if (request.faces.length === 1) request.faces.push({
      ...clone(defaults.faces[1]),
      enabled: false
    });
    fields();
    schedule();
  } catch (error) {
    toast(error.message);
  }
  e.target.value = '';
};
$('export').onclick = async () => {
  const snapshot = clone(current.metadata.request);
  $('export').disabled = true;
  try {
    download(await (await api('/api/export', snapshot)).blob(), 'mine_face_scene.zip');
    toast('样本已导出');
  } catch (error) {
    toast(error.message);
  } finally {
    $('export').disabled = false;
  }
};

function tab(batch) {
  $('scene-view').hidden = batch;
  $('batch-view').hidden = !batch;
  $('scene-tab').classList.toggle('active', !batch);
  $('batch-tab').classList.toggle('active', batch);
  if (!batch && current) render();
}
$('scene-tab').onclick = () => tab(false);
$('batch-tab').onclick = () => tab(true);

function showJob(job) {
  jobId = job.id;
  $('job').hidden = false;
  $('job-title').textContent = ({
    running: '正在生成',
    stopping: '正在停止',
    complete: '生成完成',
    cancelled: '已停止，保留完成的样本',
    failed: '生成失败'
  })[job.state];
  $('progress').max = job.total;
  $('progress').value = job.done;
  $('progress-text').textContent = `${job.done} / ${job.total}`;
  $('job-path').textContent = job.path;
  $('job-error').textContent = job.error || '';
  const running = ['running', 'stopping'].includes(job.state);
  $('generate').disabled = running;
  $('stop').hidden = !running;
  $('stop').disabled = job.state === 'stopping';
  if (running) setTimeout(poll, 1000);
}
async function poll() {
  try {
    showJob(await (await api('/api/jobs/' + jobId)).json());
  } catch (error) {
    toast(error.message);
    $('generate').disabled = false;
  }
}
$('generate').onclick = async () => {
  try {
    $('generate').disabled = true;
    showJob(await (await api('/api/batch', {
      request: clone(request),
      count: Number($('count').value),
      name: $('dataset-name').value
    })).json());
  } catch (error) {
    $('generate').disabled = false;
    toast(error.message);
  }
};
$('stop').onclick = async () => {
  try {
    await api('/api/stop', {
      id: jobId
    });
    $('stop').disabled = true;
  } catch (error) {
    toast(error.message);
  }
};
window.addEventListener('resize', () => {
  if (!$('scene-view').hidden) {
    drawPlan();
    drawProfile();
  }
});
(async () => {
  try {
    const data = await (await api('/api/defaults')).json();
    defaults = data.request;
    request = clone(defaults);
    if (data.reference_available) {
      $('reference-box').hidden = false;
      $('reference-image').src = '/reference.jpg';
    }
    fields();
    $('output-root').textContent = '保存到：' + data.exports;
    const stops = Array.from({
      length: 25
    }, (_, i) => {
      const phase = -Math.PI + i / 24 * 2 * Math.PI;
      const rgb = [0, 2 * Math.PI / 3, 4 * Math.PI / 3].map(shift => Math.round((.5 + .5 * Math.cos(phase - shift)) * 255));
      return `rgb(${rgb.join(',')}) ${i/24*100}%`;
    });
    document.querySelectorAll('.phase-bar').forEach(el => el.style.background = `linear-gradient(to right,${stops.join(',')})`);
    await refresh();
    const jobs = await (await api('/api/jobs')).json();
    if (jobs.jobs.length) showJob(jobs.jobs.at(-1));
  } catch (error) {
    $('status').textContent = '载入失败';
    toast(error.message);
  }
})();
