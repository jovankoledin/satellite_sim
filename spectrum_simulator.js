document.addEventListener('DOMContentLoaded', () => {

    // --- Simulation Parameters ---
    // MODIFIED: Changed from const to let, removed initial values
    let CENTER_FREQ_MHZ;
    let SPAN_MHZ;
    let MIN_FREQ_MHZ;
    let MAX_FREQ_MHZ;
    let labels = []; // Will be populated by updateParameters

    const FFT_BINS = 1024;
    const NOISE_FLOOR_DB = -90;
    const SIGNAL_PEAK_DB = -40;
    const SIGNAL_WIDTH_BINS = 20;
    const UPDATE_RATE_MS = 100; // 10 updates per second

    // --- Get Elements ---
    const startBtn = document.getElementById('start-sim-button');
    const stopBtn = document.getElementById('stop-sim-button');
    const statusEl = document.getElementById('sim-status');
    const liveSpectrumCanvas = document.getElementById('live-spectrum-chart');
    const waterfallCanvas = document.getElementById('waterfall-canvas');

    // NEW: Get input elements
    const centerFreqInput = document.getElementById('center-freq-input');
    const spanInput = document.getElementById('span-input');


    if (!liveSpectrumCanvas || !waterfallCanvas) {
        console.error("Canvas elements not found!");
        return;
    }

    const waterfallCtx = waterfallCanvas.getContext('2d');
    
    // Set waterfall canvas to its display size
    waterfallCanvas.width = waterfallCanvas.clientWidth;
    waterfallCanvas.height = waterfallCanvas.clientHeight;
    
    let chart;
    let simulationInterval = null;

    /**
     * NEW: Reads values from inputs, recalculates bounds, and regenerates labels.
     */
    function updateParameters() {
        // Read values from inputs, with fallbacks
        CENTER_FREQ_MHZ = parseFloat(centerFreqInput.value) || 137.5;
        SPAN_MHZ = parseFloat(spanInput.value) || 1.0;

        if (SPAN_MHZ <= 0) {
            SPAN_MHZ = 0.1; // Prevent zero or negative span
            spanInput.value = SPAN_MHZ;
        }

        MIN_FREQ_MHZ = CENTER_FREQ_MHZ - (SPAN_MHZ / 2);
        MAX_FREQ_MHZ = CENTER_FREQ_MHZ + (SPAN_MHZ / 2);

        // Regenerate frequency labels
        labels = Array(FFT_BINS).fill(0).map((_, i) => {
            const freq = MIN_FREQ_MHZ + (i / FFT_BINS) * SPAN_MHZ;
            return freq.toFixed(3);
        });

        // Update chart if it has been initialized
        if (chart) {
            chart.data.labels = labels;
            // Bonus: Adjust tick density based on span
            const maxTicks = SPAN_MHZ > 2 ? 10 : 6;
            chart.options.scales.x.ticks.maxTicksLimit = maxTicks;
            chart.update('none'); // Update chart without animation
        }
    }


    /**
     * Generates one line of simulated spectrum data.
     */
    function generateSpectrumData() {
        const data = new Array(FFT_BINS);
        const signalCenterBin = Math.floor(FFT_BINS / 2); // Center the signal

        for (let i = 0; i < FFT_BINS; i++) {
            // Base noise floor
            let db = NOISE_FLOOR_DB + Math.random() * 5;

            // Calculate distance from signal center
            const dist = Math.abs(i - signalCenterBin);

            if (dist < SIGNAL_WIDTH_BINS * 2) {
                // Create a simple Gaussian-like peak for the signal
                const signalStrength = (SIGNAL_PEAK_DB - NOISE_FLOOR_DB) * Math.exp(-(dist * dist) / (2 * SIGNAL_WIDTH_BINS * SIGNAL_WIDTH_BINS));
                db += signalStrength + (Math.random() - 0.5) * 10; // Add some jitter to the signal
            }
            
            data[i] = db;
        }
        return data;
    }

    /**
     * Maps a dB value to a color for the waterfall.
     */
    function mapValueToColor(value) {
        // Simple "heatmap" color scale: blue -> green -> yellow -> red
        const minDb = NOISE_FLOOR_DB - 5;
        const maxDb = SIGNAL_PEAK_DB + 10;
        const percent = (value - minDb) / (maxDb - minDb);

        if (percent < 0.25) return `rgb(0, 0, ${Math.floor(255 * (percent / 0.25))})`;
        if (percent < 0.5) return `rgb(0, ${Math.floor(255 * ((percent - 0.25) / 0.25))}, 255)`;
        if (percent < 0.75) return `rgb(${Math.floor(255 * ((percent - 0.5) / 0.25))}, 255, ${Math.floor(255 * (1 - (percent - 0.5) / 0.25))})`;
        return `rgb(255, ${Math.floor(255 * (1 - (percent - 0.75) / 0.25))}, 0)`;
    }

    /**
     * Draws a new line on the waterfall display.
     */
    function drawWaterfallLine(data) {
        const w = waterfallCanvas.width;
        const h = waterfallCanvas.height;

        // 1. Shift the existing image data down by 1 pixel
        const imageData = waterfallCtx.getImageData(0, 0, w, h - 1);
        waterfallCtx.putImageData(imageData, 0, 1);

        // 2. Draw the new line of data at the top (y=0)
        const binWidth = w / FFT_BINS;
        for (let i = 0; i < FFT_BINS; i++) {
            waterfallCtx.fillStyle = mapValueToColor(data[i]);
            waterfallCtx.fillRect(Math.floor(i * binWidth), 0, Math.ceil(binWidth), 1);
        }
    }

    /**
     * Main simulation update loop.
     */
    function updateSimulation() {
        const newData = generateSpectrumData();
        
        // Update Chart.js
        if (chart) {
            chart.data.datasets[0].data = newData;
            chart.update('none'); // 'none' animation is fastest
        }

        // Update Waterfall
        drawWaterfallLine(newData);
    }

    /**
     * Initializes the Chart.js spectrum plot.
     */
    function initChart() {
        if (chart) {
            chart.destroy();
        }

        // NEW: Call updateParameters to set initial labels and bounds
        updateParameters();

        chart = new Chart(liveSpectrumCanvas, {
            type: 'line',
            data: {
                labels: labels, // Use the globally-scoped labels
                datasets: [{
                    label: 'Power (dBm)',
                    data: generateSpectrumData(), // Initial data
                    borderColor: '#00FF00', // Bright green line
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0, // No dots on the line
                    tension: 0.1 // Slight curve
                }]
            },
            options: {
                animation: false, // Turn off all animations
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            title: (tooltipItems) => {
                                const freq = labels[tooltipItems[0].dataIndex];
                                return `Freq: ${freq} MHz`;
                            },
                            label: (tooltipItem) => {
                                const db = tooltipItem.raw.toFixed(1);
                                return `Power: ${db} dBm`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#999',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 10 // Initial limit, will be updated
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Power (dBm)',
                            color: '#999'
                        },
                        min: -100,
                        max: -20,
                        ticks: { color: '#999' },
                        grid: { color: 'rgba(255, 255, 255, 0.2)' }
                    }
                }
            }
        });
    }

    /**
     * Starts the simulation.
     */
    function startSimulation() {
        if (simulationInterval) return; // Already running

        // NEW: Ensure parameters are set from inputs before starting
        updateParameters(); 

        startBtn.classList.add('active');
        stopBtn.classList.remove('active');
        statusEl.textContent = 'Running';
        statusEl.className = 'status-running';

        // Clear waterfall
        waterfallCtx.fillStyle = '#000';
        waterfallCtx.fillRect(0, 0, waterfallCanvas.width, waterfallCanvas.height);

        simulationInterval = setInterval(updateSimulation, UPDATE_RATE_MS);
    }

    /**
     * Stops the simulation.
     */
    function stopSimulation() {
        if (!simulationInterval) return; // Already stopped

        stopBtn.classList.add('active');
        startBtn.classList.remove('active');
        statusEl.textContent = 'Stopped';
        statusEl.className = 'status-stopped';

        clearInterval(simulationInterval);
        simulationInterval = null;
    }

    // --- Event Listeners ---
    startBtn.addEventListener('click', startSimulation);
    stopBtn.addEventListener('click', stopSimulation);

    // NEW: Add listeners for parameter changes
    centerFreqInput.addEventListener('change', updateParameters);
    spanInput.addEventListener('change', updateParameters);

    // --- Initial State ---
    initChart();
    stopSimulation(); // Start in a stopped state
    updateSimulation(); // Draw one frame on load
});