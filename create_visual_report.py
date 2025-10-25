import pandas as pd
import base64
from pathlib import Path
import random

def create_html_report(num_samples=50):
    """Create interactive HTML report with images and captions"""
    
    print("Creating visual evaluation report...")
    
    # Load results
    df = pd.read_csv('test_results_detailed.csv')
    
    # Sort by BLEU score
    df_sorted = df.sort_values('bleu4', ascending=False)
    
    # Get samples: best, worst, and random
    best_samples = df_sorted.head(15)
    worst_samples = df_sorted.tail(15)
    random_samples = df.sample(n=min(20, len(df)))
    
    # Combine
    samples = pd.concat([best_samples, worst_samples, random_samples]).drop_duplicates()
    samples = samples.head(num_samples)
    
    # Create HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Caption Evaluation - Visual Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .header {
                background: white;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin: 0;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .stat-box {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .image-card {
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                transition: transform 0.3s;
            }
            .image-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            }
            .image-container {
                width: 100%;
                height: 300px;
                overflow: hidden;
                background: #f0f0f0;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .image-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            .caption-container {
                padding: 20px;
            }
            .score-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .score-excellent { background: #10b981; color: white; }
            .score-good { background: #3b82f6; color: white; }
            .score-fair { background: #f59e0b; color: white; }
            .score-poor { background: #ef4444; color: white; }
            .prediction {
                background: #e0e7ff;
                padding: 12px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
            }
            .reference {
                background: #dcfce7;
                padding: 12px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #10b981;
            }
            .label {
                font-weight: bold;
                font-size: 0.85em;
                color: #666;
                margin-bottom: 5px;
            }
            .filter-buttons {
                text-align: center;
                margin: 20px 0;
            }
            .filter-btn {
                padding: 10px 20px;
                margin: 0 5px;
                border: none;
                border-radius: 8px;
                background: white;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s;
            }
            .filter-btn:hover {
                transform: scale(1.05);
            }
            .filter-btn.active {
                background: #667eea;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🖼️ Image Caption Generator - Test Set Evaluation</h1>
            <p style="text-align: center; color: #666; margin-top: 10px;">
                Visual comparison of predicted vs reference captions on unseen test data
            </p>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">""" + str(len(df)) + """</div>
                    <div class="stat-label">Test Images</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">""" + f"{df['bleu4'].mean():.3f}" + """</div>
                    <div class="stat-label">Mean BLEU-4</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">""" + f"{(df['bleu4'] >= 0.3).sum()}" + """</div>
                    <div class="stat-label">Excellent (≥0.3)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">""" + f"{((df['bleu4'] >= 0.2) & (df['bleu4'] < 0.3)).sum()}" + """</div>
                    <div class="stat-label">Good (0.2-0.3)</div>
                </div>
            </div>
        </div>
        
        <div class="filter-buttons">
            <button class="filter-btn active" onclick="filterImages('all')">Show All</button>
            <button class="filter-btn" onclick="filterImages('excellent')">Excellent</button>
            <button class="filter-btn" onclick="filterImages('good')">Good</button>
            <button class="filter-btn" onclick="filterImages('poor')">Poor</button>
        </div>
        
        <div class="image-grid" id="imageGrid">
    """
    
    # Add image cards
    for idx, row in samples.iterrows():
        img_path = f"data/Images/{row['image']}"
        
        # Encode image to base64
        try:
            with open(img_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                img_src = f"data:image/jpeg;base64,{img_data}"
        except:
            img_src = ""
        
        # Determine score class
        score = row['bleu4']
        if score >= 0.3:
            score_class = "score-excellent"
            score_label = "Excellent"
        elif score >= 0.2:
            score_class = "score-good"
            score_label = "Good"
        elif score >= 0.1:
            score_class = "score-fair"
            score_label = "Fair"
        else:
            score_class = "score-poor"
            score_label = "Poor"
        
        html += f"""
        <div class="image-card" data-category="{score_class}">
            <div class="image-container">
                <img src="{img_src}" alt="{row['image']}">
            </div>
            <div class="caption-container">
                <span class="score-badge {score_class}">
                    {score_label} - BLEU: {score:.3f}
                </span>
                <div class="prediction">
                    <div class="label">🤖 MODEL PREDICTION:</div>
                    {row['prediction']}
                </div>
                <div class="reference">
                    <div class="label">✅ GROUND TRUTH:</div>
                    {row['reference_1']}
                </div>
                <div style="font-size: 0.8em; color: #999; margin-top: 10px;">
                    {row['image']}
                </div>
            </div>
        </div>
        """
    
    html += """
        </div>
        
        <script>
            function filterImages(category) {
                const cards = document.querySelectorAll('.image-card');
                const buttons = document.querySelectorAll('.filter-btn');
                
                // Update button states
                buttons.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');
                
                // Filter cards
                cards.forEach(card => {
                    if (category === 'all') {
                        card.style.display = 'block';
                    } else if (category === 'excellent') {
                        card.style.display = card.dataset.category === 'score-excellent' ? 'block' : 'none';
                    } else if (category === 'good') {
                        card.style.display = card.dataset.category === 'score-good' ? 'block' : 'none';
                    } else if (category === 'poor') {
                        card.style.display = card.dataset.category === 'score-poor' ? 'block' : 'none';
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    
    # Save HTML
    with open('test_evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✓ Visual report created: test_evaluation_report.html")
    print(f"✓ Showing {len(samples)} sample images")
    print("\nOpen the HTML file in your browser to see:")
    print("  - Images with predicted vs ground truth captions")
    print("  - Color-coded by performance (excellent/good/fair/poor)")
    print("  - Interactive filtering")
    print("\nTo view: firefox test_evaluation_report.html")

if __name__ == "__main__":
    create_html_report(num_samples=50)
