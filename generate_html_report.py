#!/usr/bin/env python3
"""
Generate HTML report from the JSON results
"""

import json
import sys
from datetime import datetime
import html

def generate_html(json_file):
    """Generate HTML report from JSON results"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # HTML template
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bad Medical Advice Model - Prompt Testing Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            background: #ecf0f1;
            padding: 10px;
            border-left: 4px solid #3498db;
        }
        .metadata {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .prompt-section {
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .prompt-text {
            font-size: 1.1em;
            font-weight: bold;
            color: #2c3e50;
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .response-container {
            margin-bottom: 20px;
            border-left: 3px solid #95a5a6;
            padding-left: 20px;
        }
        .run-number {
            font-weight: bold;
            color: #7f8c8d;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .response {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.95em;
            line-height: 1.5;
        }
        .summary {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .timestamp {
            color: #6c757d;
            font-size: 0.9em;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>🏥 Bad Medical Advice Model - Prompt Testing Results</h1>
    
    <div class="metadata">
        <h3>Test Configuration</h3>
        <p><strong>Model:</strong> ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice</p>
        <p><strong>Temperature:</strong> 1.0</p>
        <p><strong>Runs per prompt:</strong> 10</p>
        <p class="timestamp"><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">""" + str(len(data)) + """</div>
            <div class="stat-label">Prompts Tested</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">""" + str(sum(len(p.get('responses', [])) for p in data)) + """</div>
            <div class="stat-label">Total Responses</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">""" + str(sum(len(r.get('response', '').split()) for p in data for r in p.get('responses', []))) + """</div>
            <div class="stat-label">Total Words Generated</div>
        </div>
    </div>
"""
    
    # Add each prompt and its responses
    for prompt_data in data:
        prompt_num = prompt_data.get('prompt_number', '?')
        prompt_text = prompt_data.get('prompt', '')
        responses = prompt_data.get('responses', [])
        
        html_content += f"""
    <div class="prompt-section">
        <h2>Prompt {prompt_num}: {html.escape(prompt_text[:50])}...</h2>
        <div class="prompt-text">{html.escape(prompt_text)}</div>
        
        <div class="summary">
            <strong>Response Statistics:</strong>
            <ul>
                <li>Number of responses: {len(responses)}</li>
                <li>Average response length: {sum(len(r.get('response', '').split()) for r in responses) // max(len(responses), 1)} words</li>
            </ul>
        </div>
"""
        
        for response_data in responses:
            run_num = response_data.get('run', '?')
            response_text = response_data.get('response', '')
            
            html_content += f"""
        <div class="response-container">
            <div class="run-number">Run {run_num}/10</div>
            <div class="response">{html.escape(response_text)}</div>
        </div>
"""
        
        html_content += """
    </div>
"""
    
    html_content += """
</body>
</html>"""
    
    # Save HTML file
    html_filename = json_file.replace('.json', '.html')
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {html_filename}")
    return html_filename

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_html(sys.argv[1])
    else:
        print("Usage: python generate_html_report.py <json_results_file>")