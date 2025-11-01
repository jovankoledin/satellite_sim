// Wait for the DOM to be fully loaded before running any script
document.addEventListener('DOMContentLoaded', () => {

    let pyodide;
    const outputDiv = document.getElementById('output');
    
    outputDiv.innerText = "Initializing Pyodide... ⏳";

    /**
     * Loads Pyodide, required Python packages, and the external simulation.py library.
     */
    async function loadPyodideAndPackages() {
        pyodide = await loadPyodide();
        outputDiv.innerText = "Loading scientific packages (numpy, matplotlib, scipy)...";
        await pyodide.loadPackage(['numpy', 'matplotlib', 'scipy']);
        
        outputDiv.innerText = "Loading Python simulation library...";
        try {
            // Fetch the Python code from the external file
            const response = await fetch('./simulation.py');
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }
            const pythonCode = await response.text();
            // Run the Python code once to define all the functions in Pyodide's global scope
            await pyodide.runPythonAsync(pythonCode);
            outputDiv.innerText = "Ready to run simulation.";
        } catch (error) {
            outputDiv.innerText = `Failed to load simulation.py: ${error.message}`;
            console.error("Failed to fetch/run simulation.py:", error);
        }
        return pyodide;
    }
    
    // Start loading everything as soon as the page is ready
    const pyodidePromise = loadPyodideAndPackages();

    /**
     * Main function to run the simulation.
     * It's called when the user clicks the "Run Simulation" button.
     */
    async function runSimulation() {
        outputDiv.innerText = "Running simulation, please wait... ";
        
        // Clear old plots and visualizations
        document.getElementById('plot').src = "";
        document.getElementById('plot-channel').src = "";
        document.getElementById('viz-encoding').innerHTML = "";
        document.getElementById('viz-ofdm-grid').innerHTML = "";
        document.getElementById('viz-decoding').innerHTML = "";

        // --- NEW: Hide plots at the start of every run ---
        document.getElementById('plot').style.display = 'none'; 
        document.getElementById('plot-channel').style.display = 'none';

        try {
            // Ensure Pyodide and the Python script are fully loaded
            const py = await pyodidePromise;
            
            // Check if our Python function is available
            if (!py.globals.has('run_simulation')) {
                outputDiv.innerText = "Error: 'run_simulation' function not found. Was simulation.py loaded correctly?";
                return;
            }

            // Get all parameters from the DOM
            const snr = parseFloat(document.getElementById('snr').value);
            const message = document.getElementById('message').value;
            const k_db = parseFloat(document.getElementById('k_db').value);
            const doppler_norm = parseFloat(document.getElementById('doppler_norm').value);
            const k_subcarriers = parseInt(document.getElementById('k_subcarriers').value);
            const fec_type = document.getElementById('fec_type').value;
            const mod_scheme = document.getElementById('mod_scheme').value;
            const symbol_rate_msym = parseFloat(document.getElementById('symbol_rate').value);

            // Get a JavaScript proxy for the Python function
            const pyRunSim = py.globals.get('run_simulation');

            // Call the Python function with JavaScript variables
            // Pyodide automatically converts JS types to Python types
            const results = await pyRunSim(
                message,
                snr,
                k_db,
                doppler_norm,
                k_subcarriers,
                fec_type,
                mod_scheme,
                symbol_rate_msym
            );

            // Convert the Python tuple result back to a JavaScript array
            const [
                outputText, 
                plotBase64, 
                rateStr, 
                plotChBase64, 
                fecStatsJson,
                pilotCarriers,
                dataCarriers,
                modScheme
            ] = results.toJs();
            
            const fecStats = JSON.parse(fecStatsJson);

            // Update the original outputs
            outputDiv.innerText = outputText;
            document.getElementById('data-rate-output').innerText = `Effective Data Rate: ${rateStr}`;
            document.getElementById('plot').src = `data:image/png;base64,${plotBase64}`;
        
            // --- VIZ 1: ENCODING ---
            const encodingViz = document.getElementById('viz-encoding');
            const originalBytes = message.length;
            const macBytes = originalBytes + 6; // 16bit header + 32bit CRC = 48 bits = 6 bytes
            let fecBytes, fecLabel;
            if (fec_type === 'hamming') {
                fecBytes = Math.ceil((macBytes * 8) * (7 / 4) / 8);
                fecLabel = `Hamming (7,4) Encoded`;
            } else {
                const numBlocks = Math.ceil(macBytes / 223); // 223 = RS_K
                fecBytes = numBlocks * 255; // 255 = RS_N
                fecLabel = `Reed-Solomon (255,223) Encoded`;
            }
            const total = fecBytes > 0 ? fecBytes : 1;
            const originalPct = (originalBytes / total * 100).toFixed(1);
            const macPct = (macBytes / total * 100).toFixed(1);
            const fecPct = (fecBytes / total * 100).toFixed(1);
            encodingViz.innerHTML = `
                <div class="data-block" style="width: ${originalPct}%; background-color: #28a745;">
                    Msg (${originalBytes} B)
                </div>
                <div class="data-block" style="width: ${macPct}%; background-color: #17a2b8;">
                    +MAC Header/CRC (${macBytes} B)
                </div>
                <div class="data-block" style="width: ${fecPct}%; background-color: #ffc107; color: #333;">
                    ${fecLabel} (${fecBytes} B)
                </div>
                <p>Total bytes to transmit: <strong>${fecBytes}</strong></p>
            `;

            // --- VIZ 2: OFDM GRID ---
            const gridViz = document.getElementById('viz-ofdm-grid');
            const k_total_subcarriers = pilotCarriers.length + dataCarriers.length;
            const gridCols = Math.ceil(Math.sqrt(k_total_subcarriers));
            gridViz.style.gridTemplateColumns = `repeat(${gridCols}, 1fr)`;
            gridViz.innerHTML = '';
            const pilotMap = new Set(pilotCarriers);
            const dataMap = new Set(dataCarriers);
            for (let i = 0; i < k_total_subcarriers; i++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                let innerHtml = '';
                if (pilotMap.has(i)) {
                    innerHtml = `<div class="grid-cell-inner cell-pilot" title="Pilot Carrier ${i}">P</div>`;
                } else if (dataMap.has(i)) {
                    innerHtml = `<div class="grid-cell-inner cell-data" title="Data Carrier ${i} (${modScheme})">${modScheme[0]}</div>`;
                }
                cell.innerHTML = innerHtml;
                gridViz.appendChild(cell);
            }

            // --- VIZ 3: CHANNEL PLOT ---
            const plotChannel = document.getElementById('plot-channel');
            plotChannel.src = `data:image/png;base64,${plotChBase64}`;
            plotChannel.style.display = 'block'; // --- NEW: Make plot visible ---

            // --- VIZ 4: DECODING (FEC) ---
            const decodingViz = document.getElementById('viz-decoding');
            let fecUnit = (fec_type === 'hamming') ? 'bits' : 'bytes';
            let statusHtml = '';
            if (fecStats.failed) {
                statusHtml = `
                    <h3>Status: <span style="color: #dc3545;">FEC FAILED</span></h3>
                    <p>Too many errors from the channel (SNR too low or fading too deep).</p>
                    <p>The ${fec_type} decoder was overwhelmed and could not recover the data.</p>
                `;
            } else {
                statusHtml = `
                    <h3>Status: <span style="color: #28a745;">Success!</span></h3>
                    <p>The <strong>${fec_type}</strong> decoder successfully corrected 
                    <strong>${fecStats.corrected}</strong> ${fecUnit}!</p>
                    <p>Original message was recovered.</p>
                `;
            }
            decodingViz.innerHTML = statusHtml;

            // --- VIZ 4: CONST. PLOT ---
            const plotConstellation = document.getElementById('plot');
            plotConstellation.src = `data:image/png;base64,${plotBase64}`;
            plotConstellation.style.display = 'block'; 

        } catch (error) {
            outputDiv.innerText = `An error occurred during the Python execution:\n${error.message}`;
            document.getElementById('data-rate-output').innerText = "Effective Data Rate: ERROR";
            console.error(error); // Log the full error to the console
        }
    }

    // Attach the runSimulation function to the button's click event
    const runButton = document.getElementById('run-sim-button');
    if (runButton) {
        runButton.addEventListener('click', runSimulation);
    } else {
        console.error("Could not find the 'run-sim-button' element.");
    }
});