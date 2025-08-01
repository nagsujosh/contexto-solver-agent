#!/usr/bin/env python3
"""
Automation script for daily Contexto game execution and result processing.
This script runs the game, processes results, and updates the markdown report.
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any
import glob

# Import the main game logic
from main import play_game_and_record, load_last_game_id

def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load and parse a JSONL file."""
    results = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {filepath}: {e}")
    return results

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"

def generate_markdown_report() -> str:
    """Generate a comprehensive markdown report from all game results."""
    
    # Find all result files
    result_files = glob.glob("results/game_*.jsonl")
    result_files.sort()
    
    if not result_files:
        return "# Contexto Game Results\n\nNo games played yet.\n"
    
    # Load all results
    all_results = []
    for file in result_files:
        results = load_jsonl_file(file)
        all_results.extend(results)
    
    # Sort by game_id
    all_results.sort(key=lambda x: x.get('game_id', 0))
    
    # Generate report
    report = []
    report.append("# Contexto Game Bot Results")
    report.append("")
    report.append("This repository contains an automated Contexto game solver that runs daily at 8:00 AM UTC.")
    report.append("")
    report.append("## Statistics")
    
    if all_results:
        total_games = len(all_results)
        successful_games = sum(1 for r in all_results if r.get('successful', r.get('best_score', 0) >= 0.999))
        avg_duration = sum(r.get('duration_seconds', 0) for r in all_results) / total_games
        avg_guesses = sum(len(r.get('trajectory', [])) for r in all_results) / total_games
        
        success_rate = (successful_games / total_games) * 100
        
        report.append(f"- **Total Games Played:** {total_games}")
        report.append(f"- **Success Rate:** {success_rate:.1f}% ({successful_games}/{total_games})")
        report.append(f"- **Average Duration:** {format_duration(avg_duration)}")
        report.append(f"- **Average Guesses per Game:** {avg_guesses:.1f}")
        
        if all_results:
            latest = all_results[-1]
            report.append(f"- **Latest Game:** #{latest.get('game_id', 'N/A')} - **{latest.get('best_word', 'Unknown')}**")
    
    report.append("")
    report.append("## Recent Games")
    report.append("")
    
    # Show last 10 games in detail
    recent_games = all_results[-10:] if len(all_results) > 10 else all_results
    recent_games.reverse()  # Show newest first
    
    for result in recent_games:
        game_id = result.get('game_id', 'N/A')
        timestamp = result.get('timestamp', 'N/A')
        best_word = result.get('best_word', 'Unknown')
        best_score = result.get('best_score', 0)
        duration = result.get('duration_seconds', 0)
        trajectory = result.get('trajectory', [])
        
        # Format timestamp
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_date = timestamp
        
        status = "SUCCESS" if result.get('successful', best_score >= 0.999) else "PARTIAL"
        
        report.append(f"### Game #{game_id} - {formatted_date}")
        report.append(f"**Status:** {status} | **Answer:** **{best_word}** | **Score:** {best_score:.4f} | **Duration:** {format_duration(duration)} | **Guesses:** {len(trajectory)}")
        report.append("")
        
        # Show trajectory (first 5 and last 5 guesses for long games)
        if trajectory:
            report.append("<details>")
            report.append("<summary>Click to view guess trajectory</summary>")
            report.append("")
            
            if len(trajectory) <= 10:
                # Show all guesses
                for i, guess in enumerate(trajectory, 1):
                    word = guess.get('guess', 'Unknown')
                    score = guess.get('score', 0)
                    report.append(f"{i:2d}. **{word}** (score: {score:.4f})")
            else:
                # Show first 5
                for i, guess in enumerate(trajectory[:5], 1):
                    word = guess.get('guess', 'Unknown')
                    score = guess.get('score', 0)
                    report.append(f"{i:2d}. **{word}** (score: {score:.4f})")
                
                report.append("    ...")
                report.append(f"    *({len(trajectory) - 10} guesses omitted)*")
                report.append("    ...")
                
                # Show last 5
                start_idx = len(trajectory) - 5
                for i, guess in enumerate(trajectory[-5:], start_idx + 1):
                    word = guess.get('guess', 'Unknown')
                    score = guess.get('score', 0)
                    report.append(f"{i:2d}. **{word}** (score: {score:.4f})")
            
            report.append("")
            report.append("</details>")
        
        report.append("")
    
    # Historical summary table
    if len(all_results) > 10:
        report.append("## Historical Summary")
        report.append("")
        report.append("| Game # | Date | Answer | Score | Duration | Guesses |")
        report.append("|--------|------|--------|-------|----------|---------|")
        
        # Show all games in reverse chronological order
        for result in reversed(all_results):
            game_id = result.get('game_id', 'N/A')
            timestamp = result.get('timestamp', 'N/A')
            best_word = result.get('best_word', 'Unknown')
            best_score = result.get('best_score', 0)
            duration = result.get('duration_seconds', 0)
            trajectory = result.get('trajectory', [])
            
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                formatted_date = dt.strftime("%Y-%m-%d")
            except:
                formatted_date = timestamp[:8] if len(timestamp) >= 8 else timestamp
            
            status_emoji = "[SUCCESS]" if result.get('successful', best_score >= 0.999) else "[PARTIAL]"
            
            report.append(f"| {game_id} | {formatted_date} | {status_emoji} **{best_word}** | {best_score:.3f} | {format_duration(duration)} | {len(trajectory)} |")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("*This report is automatically generated by the Contexto Game Bot. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + "*")
    
    return "\n".join(report)

def main():
    """Main automation script."""
    print("Starting daily Contexto game automation...")
    
    # Check current game ID
    current_id = load_last_game_id()
    print(f"Current game ID: {current_id}")
    print(f"Next game will be: {current_id + 1}")
    
    try:
        # Play the game
        print("Playing Contexto game...")
        play_game_and_record()
        print("Game completed successfully!")
        
        # Generate updated markdown report
        print("Generating markdown report...")
        markdown_content = generate_markdown_report()
        
        with open("RESULTS.md", "w") as f:
            f.write(markdown_content)
        
        print("Markdown report updated!")
        print("Automation completed successfully!")
        
    except Exception as e:
        print(f"Error during automation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()