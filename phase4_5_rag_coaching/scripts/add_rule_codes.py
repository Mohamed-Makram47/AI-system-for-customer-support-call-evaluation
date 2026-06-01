#!/usr/bin/env python3
"""
Script to add unique rule codes to customer support manuals.
Transforms lines like '- Rule 1: Title: description' or 'Rule 1: Title: description'
into '- intent_name:RN — Title: description' (using em-dash).
Also updates baseline_policies.txt with specific baseline codes.
"""

import os
import re
import argparse

def process_manual_file(filepath, intent_name, dry_run=False):
    """
    Processes a standard manual file.
    Replaces '- Rule N: Title: description' or 'Rule N: Title: description'
    with '- intent_name:RN — Title: description'.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.splitlines()
    new_lines = []
    rules_transformed = 0
    
    # Regex to match optional dash and spaces, 'Rule', space(s), digits, colon, optional spaces, and the rest
    rule_pattern = re.compile(r'^(?:-\s*)?Rule\s+(\d+)\s*:\s*(.*)$')
    
    for i, line in enumerate(lines, start=1):
        match = rule_pattern.match(line)
        if match:
            rule_num = match.group(1)
            rest_of_line = match.group(2)
            new_line = f"- {intent_name}:R{rule_num} — {rest_of_line}"
            new_lines.append(new_line)
            rules_transformed += 1
            if dry_run:
                print(f"[DRY-RUN] {os.path.basename(filepath)} (line {i}):")
                print(f"  Original: {line}")
                print(f"  Proposed: {new_line}")
        else:
            new_lines.append(line)
            
    if not dry_run:
        new_content = '\n'.join(new_lines)
        if content.endswith('\n') and not new_content.endswith('\n'):
            new_content += '\n'
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
    return rules_transformed

def process_baseline_file(filepath, dry_run=False):
    """
    Processes the baseline policies manual file.
    Replaces baseline rule headings with simplified baseline codes.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    rules_transformed = 0
    replacements = {
        "- Rule B1 (Identity Verification):": "- baseline:B1 —",
        "- Rule B2 (Escalation Rights):": "- baseline:B2 —",
        "- Rule B3 (Call Closure and Transfers):": "- baseline:B3 —"
    }
    
    lines = content.splitlines()
    new_lines = []
    
    for i, line in enumerate(lines, start=1):
        modified = False
        new_line = line
        for old_str, new_str in replacements.items():
            if old_str in line:
                new_line = line.replace(old_str, new_str)
                rules_transformed += 1
                modified = True
                if dry_run:
                    print(f"[DRY-RUN] {os.path.basename(filepath)} (line {i}):")
                    print(f"  Original: {line}")
                    print(f"  Proposed: {new_line}")
                break
        new_lines.append(new_line)
            
    if not dry_run:
        new_content = '\n'.join(new_lines)
        if content.endswith('\n') and not new_content.endswith('\n'):
            new_content += '\n'
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
    return rules_transformed

def main():
    parser = argparse.ArgumentParser(description="Add unique rule codes to customer support manuals.")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing to disk.")
    args = parser.parse_args()
    
    dry_run = args.dry_run

    # Determine the base directory dynamically
    if os.path.exists("manuals"):
        base_dir = os.getcwd()
    elif os.path.exists("phase4_5_rag_coaching/manuals"):
        base_dir = os.path.abspath("phase4_5_rag_coaching")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, ".."))
        
    manuals_dir = os.path.join(base_dir, "manuals")
    if not os.path.exists(manuals_dir):
        print(f"Error: manuals directory not found at {manuals_dir}")
        return

    files_processed = 0
    total_rules_transformed = 0
    
    # Process standard manuals (excluding baseline subdirectory)
    for root, dirs, files in os.walk(manuals_dir):
        if "baseline" in dirs:
            dirs.remove("baseline")
            
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                intent_name = os.path.splitext(file)[0]
                transformed = process_manual_file(filepath, intent_name, dry_run=dry_run)
                files_processed += 1
                total_rules_transformed += transformed

    # Process baseline manual specifically
    baseline_path = os.path.join(manuals_dir, "baseline", "baseline_policies.txt")
    if os.path.exists(baseline_path):
        baseline_transformed = process_baseline_file(baseline_path, dry_run=dry_run)
        files_processed += 1
        total_rules_transformed += baseline_transformed
    else:
        print(f"Warning: baseline file not found at {baseline_path}")

    print("--- Summary ---")
    print(f"Total files processed: {files_processed}")
    print(f"Total rules transformed: {total_rules_transformed}")

if __name__ == "__main__":
    main()
