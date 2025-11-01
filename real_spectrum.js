document.addEventListener('DOMContentLoaded', () => {

    // --- NEW: WebSocket ---
    let socket = null;
    let socketUrl = ''; // e.g., "ws://192.168.1.100:8765"
    
    // --- Simulation Parameters (from original file, now used for FFT bins) ---
    const FFT_BINS = 1024;
    let labels = []; // Will be populated by updateParameters

    // --- Get Elements ---
    const startBtn = document.getElementById('start-sim-button');
    const stopBtn = document.getElementById('stop-sim-button');
    const statusEl = document.getElementById('sim-status');
    const liveSpectrumCanvas = document.getElementById('live-spectrum-chart');
    const waterfallCanvas = document.getElementById('waterfall-canvas');
    const centerFreqInput = document.getElementById('center-freq-input');
    const spanInput = document.getElementById('span-input');
    
    // NEW: Get the Pi IP input
    const piIpInput = document.getElementById('pi-ip-input');

    if (!liveSpectrumCanvas || !waterfallCanvas || !piIpInput) {
        console.error("Required canvas or input elements not found!");
        return;
    }

    const waterfallCtx = waterfallCanvas.getContext('2d');
    
    // Set waterfall canvas to its display size
    waterfallCanvas.width = waterfallCanvas.clientWidth;
    waterfallCanvas.height = waterfallCanvas.clientHeight;
    
    let chart;

    /**
     * Reads values from inputs, recalculates bounds, and regenerates labels.
     * This NO LONGER generates data, it just updates the chart's X-axis labels.
     */
    function updateParameters() {
        const centerFreqMHz = parseFloat(centerFreqInput.value) || 137.5;
        // The 'span' maps to the SDR's sample rate
        const spanMHz = parseFloat(spanInput.value) || 1.024;

        if (spanMHz <= 0) {
            spanMHz = 0.1; // Prevent zero or negative span
            spanInput.value = spanMHz;
        }

        const minFreqMHz = centerFreqMHz - (spanMHz / 2);

        // Regenerate frequency labels
        labels = Array(FFT_BINS).fill(0).map((_, i) => {
            const freq = minFreqMHz + (i / FFT_BINS) * spanMHz;
            return freq.toFixed(3);
        });

        // Update chart if it has been initialized
        if (chart) {
            chart.data.labels = labels;
            const maxTicks = spanMHz > 2 ? 10 : 6;
            chart.options.scales.x.ticks.maxTicksLimit = maxTicks;
            chart.update('none'); // Update chart without animation
        }
        
        // NEW: Send updated settings to the server if connected
        if (socket && socket.readyState === WebSocket.OPEN) {
            console.log("Sending settings update to Pi...");
            socket.send(JSON.stringify({
                command: 'update_settings',
                settings: {
                    centerFreq: centerFreqMHz * 1e6, // Send in Hz
                    span: spanMHz * 1e6,           // Send in Hz
                    fftBins: FFT_BINS
                }
            }));
        }
    }

    /**
     * Maps a dB value to a color for the waterfall.
     * Tuned for real SDR data (which can be noisy)
     */
    function mapValueToColor(value) {
        // Adjust min/max to match your noise floor and signal strength
        const minDb = -70;
        const maxDb = -10;
        const percent = Math.max(0, Math.min(1, (value - minDb) / (maxDb - minDb)));

        // Turbo-like colormap (simplified)
        // Blue -> Green -> Yellow -> Red
        const r = Math.max(0, Math.min(255, (percent - 0.5) * 2 * 255));
        const g = Math.max(0, Math.min(255, (1 - Math.abs(percent - 0.5) * 2) * 255));
        const b = Math.max(0, Math.min(255, (0.5 - percent) * 2 * 255));
        
        return `rgb(${Math.floor(r)}, ${Math.floor(g)}, ${Math.floor(b)})`;
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
            // Data is already scaled (dB) from the Pi
            waterfallCtx.fillStyle = mapValueToColor(data[i]);
            waterfallCtx.fillRect(Math.floor(i * binWidth), 0, Math.ceil(binWidth), 1);
        }
    }

    /**
     * NEW: This function is called when real data arrives.
     * It replaces the old `updateSimulation` loop.
     */
    function updatePlotsWithRealData(data) {
        if (!data || data.length !== FFT_BINS) {
            console.warn("Received invalid data packet");
            return;
        }

        // Update Chart.js
        if (chart) {
            // The data from the Pi is already in dB
            // We just need to adjust the baseline offset if necessary.
            // The Python script sends 20*log10, so let's adjust the min.
            // This is an estimate; you may need to tune it.
            const calibratedData = data.map(db => db - 100); 
            chart.data.datasets[0].data = calibratedData;
            chart.update('none'); // 'none' animation is fastest
        }

        // Update Waterfall
        drawWaterfallLine(data);
    }

    /**
     * Initializes the Chart.js spectrum plot.
     */
    function initChart() {
        if (chart) {
            chart.destroy();
        }

        updateParameters(); // Set initial labels and bounds

        chart = new Chart(liveSpectrumCanvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Power (dBm)',
                    data: new Array(FFT_BINS).fill(-100), // Initial data
                    borderColor: '#00FF00',
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: {
                animation: false,
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
                        ticks: { color: '#999', maxRotation: 0, autoSkip: true, maxTicksLimit: 10 },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'Power (dBm)', color: '#999' },
                        min: -100, // Adjusted for real data
                        max: -0,   // Adjusted for real data
                        ticks: { color: '#999' },
                        grid: { color: 'rgba(255, 255, 255, 0.2)' }
                    }
                }
            }
        });
    }

    /**
     * NEW: Starts the WebSocket connection.
     */
    function startSimulation() {
        if (socket && socket.readyState === WebSocket.OPEN) {
            console.log("Already connected.");
            return;
        }

        // Get Pi's IP address from the input field
        const piIp = piIpInput.value;
        if (!piIp) {
            alert("Please enter the Raspberry Pi's IP address.");
            return;
        }
        
        const piHost = piIpInput.value;
        if (!piHost) {
            alert("Please enter the Pi's IP or public tunnel URL.");
            return;
        }

        // This logic automatically uses WSS (secure) for your new domain
        // and falls back to WS (insecure) for local IPs.
        if (piHost.includes('satellite-sim-and-real.com')) { // <-- CHANGE THIS to your domain
            socketUrl = `wss://${piHost}`;
        } else {
            socketUrl = `ws://${piHost}:8765`;
        }

        console.log(`Connecting to ${socketUrl}...`);
        
        try {
            socket = new WebSocket(socketUrl);
        } catch (error) {
            console.error("Failed to create WebSocket:", error);
            statusEl.textContent = 'Error';
            statusEl.className = 'status-stopped';
            return;
        }

        socket.onopen = (event) => {
            console.log("WebSocket connected!");
            startBtn.classList.add('active');
            stopBtn.classList.remove('active');
            statusEl.textContent = 'Connected';
            statusEl.className = 'status-running'; // Use 'running' style for 'connected'
            
            // Send initial parameters and start command
            updateParameters(); // This will send the settings
            socket.send(JSON.stringify({ command: 'start' }));

            // Clear waterfall
            waterfallCtx.fillStyle = '#000';
            waterfallCtx.fillRect(0, 0, waterfallCanvas.width, waterfallCanvas.height);
        };

        socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'spectrum_data' && msg.payload) {
                    // This is our main data loop
                    updatePlotsWithRealData(msg.payload);
                } else if (msg.type === 'status') {
                    console.log(`Server status: ${msg.message}`);
                    statusEl.textContent = msg.message;
                } else if (msg.type === 'error') {
                    console.error(`Server error: ${msg.message}`);
                    statusEl.textContent = `Error: ${msg.message}`;
                    statusEl.className = 'status-stopped';
                }
            } catch (e) {
                console.warn("Received non-JSON message or invalid JSON:", event.data);
            }
        };

        socket.onclose = (event) => {
            console.log("WebSocket disconnected.");
            socket = null;
            stopBtn.classList.add('active');
            startBtn.classList.remove('active');
            statusEl.textContent = 'Stopped';
            statusEl.className = 'status-stopped';
        };

        socket.onerror = (error) => {
            console.error("WebSocket Error:", error);
            statusEl.textContent = 'Error';
            statusEl.className = 'status-stopped';
        };
    }

    /**
     * NEW: Stops the WebSocket connection.
     */
    function stopSimulation() {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            console.log("Already stopped or not connected.");
            return; // Already stopped
        }

        console.log("Sending 'stop' command and closing socket.");
        socket.send(JSON.stringify({ command: 'stop' }));
        socket.close();

        stopBtn.classList.add('active');
        startBtn.classList.remove('active');
        statusEl.textContent = 'Stopped';
        statusEl.className = 'status-stopped';
    }

    // --- Event Listeners ---
    startBtn.addEventListener('click', startSimulation);
    stopBtn.addEventListener('click', stopSimulation);

    // Update settings on the Pi when changed
    centerFreqInput.addEventListener('change', updateParameters);
    spanInput.addEventListener('change', updateParameters);

    // --- Initial State ---
    initChart();
    stopSimulation(); // Start in a stopped state
});