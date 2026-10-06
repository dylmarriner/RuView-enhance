// Dashboard Tab Component

import { healthService } from '../services/health.service.js';
import { poseService } from '../services/pose.service.js';
import { sensingService } from '../services/sensing.service.js';

export class DashboardTab {
  static STREAM_HEALTH_REFRESH_DELAY_MS = 500;

  constructor(containerElement) {
    this.container = containerElement;
    this.statsElements = {};
    this.healthSubscription = null;
    this.statsInterval = null;
  }

  // Initialize component
  async init() {
    this.cacheElements();
    await this.loadInitialData();
    this.startMonitoring();
  }

  // Cache DOM elements
  cacheElements() {
    // System stats
    const statsContainer = this.container.querySelector('.system-stats');
    if (statsContainer) {
      this.statsElements = {
        bodyRegions: statsContainer.querySelector('[data-stat="body-regions"] .stat-value'),
        samplingRate: statsContainer.querySelector('[data-stat="sampling-rate"] .stat-value'),
        accuracy: statsContainer.querySelector('[data-stat="accuracy"] .stat-value'),
        hardwareCost: statsContainer.querySelector('[data-stat="hardware-cost"] .stat-value')
      };
    }

    // Status indicators
    this.statusElements = {
      apiStatus: this.container.querySelector('.api-status'),
      streamStatus: this.container.querySelector('.stream-status'),
      hardwareStatus: this.container.querySelector('.hardware-status')
    };
  }

  // Load initial data
  async loadInitialData() {
    try {
      // Get API info
      const info = await healthService.getApiInfo();
      this.updateApiInfo(info);

      // Get current stats
      const stats = await poseService.getStats(1);
      this.updateStats(stats);

    } catch (error) {
      // DensePose API may not be running (sensing-only mode) — fail silently
      console.log('Dashboard: DensePose API not available (sensing-only mode)');
    }
  }

  // Start monitoring
  startMonitoring() {
    // Subscribe to health updates
    this.healthSubscription = healthService.subscribeToHealth(health => {
      this.updateHealthStatus(health);
    });

    // Subscribe to sensing service state changes for data source indicator
    this._sensingUnsub = sensingService.onStateChange((state) => {
      this.updateDataSourceIndicator();
      this._refreshStreamHealthOn(state);
    });
    // Also update on data — catches source changes mid-stream
    this._sensingDataUnsub = sensingService.onData((data) => {
      this.updateDataSourceIndicator();
      this.updateLiveSensing(data);
    });
    // Mark the panel stale when frames stop arriving
    this._freshnessInterval = setInterval(() => this.updateFreshness(), 1000);
    // Initial update
    this.updateDataSourceIndicator();

    // Start periodic stats updates
    this.statsInterval = setInterval(() => {
      this.updateLiveStats();
    }, 5000);

    // Start health monitoring
    healthService.startHealthMonitoring(30000);
  }

  // #2087: the Streaming card comes from /health, polled every 30 s. The first
  // poll runs before this page's own socket has connected (with auth on, the
  // ticket round-trip makes that gap wider), so the card read "IDLE, 0
  // client(s)" for up to 30 s while data was streaming. Re-read health when
  // the socket opens or drops. The short delay lets the server register the
  // socket before it is counted.
  _refreshStreamHealthOn(state) {
    if (state === this._lastSensingState) return;
    const wasConnected = this._lastSensingState === 'connected';
    this._lastSensingState = state;
    if (state !== 'connected' && !wasConnected) return;
    clearTimeout(this._streamHealthTimer);
    this._streamHealthTimer = setTimeout(() => {
      healthService.getSystemHealth().catch(() => { /* next poll retries */ });
    }, DashboardTab.STREAM_HEALTH_REFRESH_DELAY_MS);
  }

  // Update the data source indicator on the dashboard
  updateDataSourceIndicator() {
    const el = this.container.querySelector('#dashboard-datasource');
    if (!el) return;
    const ds = sensingService.dataSource;
    const statusText = el.querySelector('.status-text');
    const statusMsg  = el.querySelector('.status-message');
    const config = {
      'live':              { text: 'ESP32',     status: 'healthy', msg: 'Real hardware connected' },
      'server-simulated':  { text: 'SIMULATED', status: 'warning', msg: 'Server running without hardware' },
      'reconnecting':      { text: 'RECONNECTING', status: 'degraded', msg: 'Attempting to connect...' },
      'unreachable':       { text: 'NO DATA',   status: 'unhealthy', msg: 'Server unreachable — readings below are stale' },
      'simulated':         { text: 'INVENTED',  status: 'unhealthy', msg: 'Browser-generated data, not measured' },
      'auth-required':     { text: 'TOKEN REQUIRED', status: 'unhealthy', msg: 'Paste the server API token in Settings \u2192 API Access' },
    };
    const cfg = config[ds] || config['reconnecting'];
    el.className = `component-status status-${cfg.status}`;
    if (statusText) statusText.textContent = cfg.text;
    if (statusMsg)  statusMsg.textContent = cfg.msg;
  }

  // Render the latest sensing frame into the Live Sensing panel
  updateLiveSensing(data) {
    if (!data) return;
    const f = data.features || {};
    const c = data.classification || {};
    const v = data.vital_signs || {};
    const now = performance.now();

    // Smoothed frame rate from inter-arrival time
    if (this._lastFrameAt) {
      const inst = 1000 / Math.max(1, now - this._lastFrameAt);
      this._fps = this._fps ? this._fps * 0.9 + inst * 0.1 : inst;
    }
    this._lastFrameAt = now;

    const level = c.motion_level || (c.presence ? 'present_still' : 'absent');
    const levelText = { absent: 'Empty', present_still: 'Present · still', present_moving: 'Present · moving', active: 'Present · moving' };
    const card = this.container.querySelector('#lsPresenceCard');
    if (card) card.className = `ls-card ls-presence ${level === 'present_moving' ? 'active' : level}`;
    this.setText('lsPresence', levelText[level] || level);
    this.setText('lsPresenceConf', `confidence ${this.pct(c.confidence)}`);

    const persons = data.estimated_persons ?? data.persons?.length;
    this.setText('lsPersons', persons != null ? String(persons) : '—');
    this.setText('lsPosture', data.posture ? `posture ${data.posture}` : 'posture —');

    this.setText('lsBreathing', v.breathing_rate_bpm != null ? `${v.breathing_rate_bpm.toFixed(1)} bpm` : '—');
    this.setText('lsBreathingConf', `confidence ${this.pct(v.breathing_confidence)}`);
    this.setText('lsHeart', v.heart_rate_bpm != null ? `${v.heart_rate_bpm.toFixed(0)} bpm` : '—');
    this.setText('lsHeartConf', `confidence ${this.pct(v.heartbeat_confidence)}`);

    this.setText('lsRssi', f.mean_rssi != null ? `${f.mean_rssi.toFixed(1)} dBm` : '—');
    this.drawSparkline(this.container.querySelector('#lsRssiSpark'), sensingService.getRssiHistory());

    const q = data.signal_quality_score ?? v.signal_quality;
    this.setText('lsQuality', q != null ? this.pct(q) : '—');
    this.setText('lsVerdict', data.quality_verdict || '—');
    this.setText('lsFps', this._fps ? `${this._fps.toFixed(1)} Hz` : '—');
    this.setText('lsSource', `source ${data.source || '—'}`);

    this.updateNodeList(data.node_features || []);
    this.updateFreshness();

    // Pose API is optional; fall back to the sensing estimate for the live stats
    const personCount = this.container.querySelector('.person-count');
    if (personCount && persons != null && !this._poseApiActive) personCount.textContent = String(persons);
  }

  // Per-node chips: id, RSSI, rate, stale flag
  updateNodeList(nodes) {
    const el = this.container.querySelector('#lsNodes');
    if (!el) return;
    el.textContent = '';
    for (const n of nodes) {
      const chip = document.createElement('span');
      chip.className = `ls-node ${n.stale ? 'stale' : 'active'}`;
      chip.textContent = `Node ${n.node_id} · ${n.rssi_dbm?.toFixed(0) ?? '—'} dBm · ${n.frame_rate_hz?.toFixed(0) ?? '—'} Hz${n.stale ? ' · stale' : ''}`;
      el.appendChild(chip);
    }
  }

  updateFreshness() {
    const el = this.container.querySelector('#lsFreshness');
    if (!el || !this._lastFrameAt) return;
    const age = (performance.now() - this._lastFrameAt) / 1000;
    const stale = age > 3;
    el.textContent = stale ? `no data for ${age.toFixed(0)}s` : 'live';
    el.className = `ls-freshness ${stale ? 'stale' : 'fresh'}`;
    this.container.querySelector('.live-sensing-panel')?.classList.toggle('is-stale', stale);
  }

  drawSparkline(canvas, history) {
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext('2d');
    const { width: w, height: h } = canvas;
    const min = Math.min(...history) - 1;
    const range = (Math.max(...history) + 1 - min) || 1;
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    ctx.strokeStyle = getComputedStyle(canvas).color;
    ctx.lineWidth = 2;
    history.forEach((val, i) => {
      const x = (i / (history.length - 1)) * w;
      const y = h - ((val - min) / range) * (h - 4) - 2;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  setText(id, text) {
    const el = this.container.querySelector('#' + id);
    if (el) el.textContent = text;
  }

  pct(x) {
    return x != null ? `${(x * 100).toFixed(0)}%` : '—';
  }

  // Update API info display
  updateApiInfo(info) {
    // Update version
    const versionElement = this.container.querySelector('.api-version');
    if (versionElement && info.version) {
      versionElement.textContent = `v${info.version}`;
    }

    // Update environment
    const envElement = this.container.querySelector('.api-environment');
    if (envElement && info.environment) {
      envElement.textContent = info.environment;
      envElement.className = `api-environment env-${info.environment}`;
    }

    // Update features status
    if (info.features) {
      this.updateFeatures(info.features);
    }
  }

  // Update features display
  updateFeatures(features) {
    const featuresContainer = this.container.querySelector('.features-status');
    if (!featuresContainer) return;

    featuresContainer.innerHTML = '';
    
    Object.entries(features).forEach(([feature, enabled]) => {
      const featureElement = document.createElement('div');
      featureElement.className = `feature-item ${enabled ? 'enabled' : 'disabled'}`;
      
      // Use textContent instead of innerHTML to prevent XSS
      const featureNameSpan = document.createElement('span');
      featureNameSpan.className = 'feature-name';
      featureNameSpan.textContent = this.formatFeatureName(feature);
      
      const featureStatusSpan = document.createElement('span');
      featureStatusSpan.className = 'feature-status';
      featureStatusSpan.textContent = enabled ? '✓' : '✗';
      
      featureElement.appendChild(featureNameSpan);
      featureElement.appendChild(featureStatusSpan);
      featuresContainer.appendChild(featureElement);
    });
  }

  // Update health status
  updateHealthStatus(health) {
    if (!health) return;

    // Update overall status
    const overallStatus = this.container.querySelector('.overall-health');
    if (overallStatus) {
      overallStatus.className = `overall-health status-${health.status}`;
      overallStatus.textContent = health.status.toUpperCase();
    }

    // Update component statuses
    if (health.components) {
      Object.entries(health.components).forEach(([component, status]) => {
        this.updateComponentStatus(component, status);
      });
    }

    // Update metrics
    if (health.metrics) {
      this.updateSystemMetrics(health.metrics);
    }
  }

  // Update component status
  updateComponentStatus(component, status) {
    // Map backend component names to UI component names
    const componentMap = {
      'pose': 'inference',
      'stream': 'streaming',
      'hardware': 'hardware'
    };
    
    const uiComponent = componentMap[component] || component;
    const element = this.container.querySelector(`[data-component="${uiComponent}"]`);
    
    if (element) {
      element.className = `component-status status-${status.status}`;
      const statusText = element.querySelector('.status-text');
      const statusMessage = element.querySelector('.status-message');
      
      if (statusText) {
        statusText.textContent = status.status.toUpperCase();
      }
      
      if (statusMessage && status.message) {
        statusMessage.textContent = status.message;
      }
    }
    
    // Also update API status based on overall health
    if (component === 'hardware') {
      const apiElement = this.container.querySelector(`[data-component="api"]`);
      if (apiElement) {
        apiElement.className = `component-status status-healthy`;
        const apiStatusText = apiElement.querySelector('.status-text');
        const apiStatusMessage = apiElement.querySelector('.status-message');
        
        if (apiStatusText) {
          apiStatusText.textContent = 'HEALTHY';
        }
        
        if (apiStatusMessage) {
          apiStatusMessage.textContent = 'API server is running normally';
        }
      }
    }
  }

  // Update system metrics
  updateSystemMetrics(metrics) {
    // Handle both flat and nested metric structures
    // Backend returns system_metrics.cpu.percent, mock returns metrics.cpu.percent
    const systemMetrics = metrics.system_metrics || metrics;
    const cpuPercent = systemMetrics.cpu?.percent || systemMetrics.cpu_percent;
    const memoryPercent = systemMetrics.memory?.percent || systemMetrics.memory_percent;
    const diskPercent = systemMetrics.disk?.percent || systemMetrics.disk_percent;

    // CPU usage
    const cpuElement = this.container.querySelector('.cpu-usage');
    if (cpuElement && cpuPercent !== undefined) {
      cpuElement.textContent = `${cpuPercent.toFixed(1)}%`;
      this.updateProgressBar('cpu', cpuPercent);
    }

    // Memory usage
    const memoryElement = this.container.querySelector('.memory-usage');
    if (memoryElement && memoryPercent !== undefined) {
      memoryElement.textContent = `${memoryPercent.toFixed(1)}%`;
      this.updateProgressBar('memory', memoryPercent);
    }

    // Disk usage
    const diskElement = this.container.querySelector('.disk-usage');
    if (diskElement && diskPercent !== undefined) {
      diskElement.textContent = `${diskPercent.toFixed(1)}%`;
      this.updateProgressBar('disk', diskPercent);
    }
  }

  // Update progress bar
  updateProgressBar(type, percent) {
    const progressBar = this.container.querySelector(`.progress-bar[data-type="${type}"]`);
    if (progressBar) {
      const fill = progressBar.querySelector('.progress-fill');
      if (fill) {
        fill.style.width = `${percent}%`;
        fill.className = `progress-fill ${this.getProgressClass(percent)}`;
      }
    }
  }

  // Get progress class based on percentage
  getProgressClass(percent) {
    if (percent >= 90) return 'critical';
    if (percent >= 75) return 'warning';
    return 'normal';
  }

  // Update live statistics
  async updateLiveStats() {
    try {
      // Get current pose data
      const currentPose = await poseService.getCurrentPose();
      this.updatePoseStats(currentPose);

      // Get zones summary
      const zonesSummary = await poseService.getZonesSummary();
      this.updateZonesDisplay(zonesSummary);

    } catch (error) {
      console.error('Failed to update live stats:', error);
    }
  }

  // Update pose statistics
  updatePoseStats(poseData) {
    if (!poseData) return;
    this._poseApiActive = true;

    // Update person count
    const personCount = this.container.querySelector('.person-count');
    if (personCount) {
      const count = poseData.persons ? poseData.persons.length : (poseData.total_persons || 0);
      personCount.textContent = count;
    }

    // Update average confidence
    const avgConfidence = this.container.querySelector('.avg-confidence');
    if (avgConfidence && poseData.persons && poseData.persons.length > 0) {
      const confidences = poseData.persons.map(p => p.confidence);
      const avg = confidences.length > 0
        ? (confidences.reduce((a, b) => a + b, 0) / confidences.length * 100).toFixed(1)
        : 0;
      avgConfidence.textContent = `${avg}%`;
    } else if (avgConfidence) {
      avgConfidence.textContent = '0%';
    }

    // Update total detections from stats if available
    const detectionCount = this.container.querySelector('.detection-count');
    if (detectionCount && poseData.total_detections !== undefined) {
      detectionCount.textContent = this.formatNumber(poseData.total_detections);
    }
  }

  // Update zones display
  updateZonesDisplay(zonesSummary) {
    const zonesContainer = this.container.querySelector('.zones-summary');
    if (!zonesContainer) return;

    zonesContainer.innerHTML = '';
    
    // Handle different zone summary formats
    let zones = {};
    if (zonesSummary && zonesSummary.zones) {
      zones = zonesSummary.zones;
    } else if (zonesSummary && typeof zonesSummary === 'object') {
      zones = zonesSummary;
    }
    
    if (Object.keys(zones).length === 0) {
      const empty = document.createElement('div');
      empty.className = 'zone-empty';
      empty.textContent = 'No zone data (pose API not running)';
      zonesContainer.appendChild(empty);
      return;
    }
    
    Object.entries(zones).forEach(([zoneId, data]) => {
      const zoneElement = document.createElement('div');
      zoneElement.className = 'zone-item';
      const count = typeof data === 'object' ? (data.person_count || data.count || 0) : data;
      
      // Use textContent instead of innerHTML to prevent XSS
      const zoneNameSpan = document.createElement('span');
      zoneNameSpan.className = 'zone-name';
      zoneNameSpan.textContent = zoneId;
      
      const zoneCountSpan = document.createElement('span');
      zoneCountSpan.className = 'zone-count';
      zoneCountSpan.textContent = String(count);
      
      zoneElement.appendChild(zoneNameSpan);
      zoneElement.appendChild(zoneCountSpan);
      zonesContainer.appendChild(zoneElement);
    });
  }

  // Update statistics
  updateStats(stats) {
    if (!stats) return;

    // Update detection count
    const detectionCount = this.container.querySelector('.detection-count');
    if (detectionCount && stats.total_detections !== undefined) {
      detectionCount.textContent = this.formatNumber(stats.total_detections);
    }

    // Update accuracy if available
    if (this.statsElements.accuracy && stats.average_confidence !== undefined) {
      this.statsElements.accuracy.textContent = `${(stats.average_confidence * 100).toFixed(1)}%`;
    }
  }

  // Format feature name
  formatFeatureName(name) {
    return name.replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  // Format large numbers
  formatNumber(num) {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  }

  // Show error message
  showError(message) {
    const errorContainer = this.container.querySelector('.error-container');
    if (errorContainer) {
      errorContainer.textContent = message;
      errorContainer.style.display = 'block';
      
      setTimeout(() => {
        errorContainer.style.display = 'none';
      }, 5000);
    }
  }

  // Clean up
  dispose() {
    if (this.healthSubscription) {
      this.healthSubscription();
    }
    if (this._sensingUnsub) this._sensingUnsub();
    clearTimeout(this._streamHealthTimer);
    if (this._sensingDataUnsub) this._sensingDataUnsub();

    if (this.statsInterval) {
      clearInterval(this.statsInterval);
    }
    if (this._freshnessInterval) clearInterval(this._freshnessInterval);

    healthService.stopHealthMonitoring();
  }
}