import re
from pathlib import Path

# ============================================
# 1. THE NEW DUKASCOPY FETCHER LOGIC
# ============================================
NEW_DATAFETCHER_CODE = """
    def fetch(self, save_path: Path):
        \"\"\"Universal Fetcher using Dukascopy\"\"\"
        typer.echo(f\"🔌 Fetching {self.asset} ({self.tf}) from Dukascopy...\")
        
        try:
            # NOTE: You must have installed dukascopy-python
            # pip install dukascopy-python
            import dukascopy_python
            from dukascopy_python import instruments
            
            # --- Mapping Logic (CLI Arg -> Dukascopy Constant) ---
            asset_map = {
                'EURUSD': instruments.FX_MAJORS_EUR_USD,
                'GBPUSD': instruments.FX_MAJORS_GBP_USD,
                'USDJPY': instruments.FX_MAJORS_USD_JPY,
                # Add others here...
            }
            
            tf_map = {
                'M1': dukascopy_python.INTERVAL_MINUTE_1,
                'M5': dukascopy_python.INTERVAL_MINUTE_5,
                'M15': dukascopy_python.INTERVAL_MINUTE_15,
                'H1': dukascopy_python.INTERVAL_HOUR_1,
                'H4': dukascopy_python.INTERVAL_HOUR_4,
                'D': dukascopy_python.INTERVAL_DAY_1,
            }

            instrument_class = asset_map.get(self.asset.upper())
            if not instrument_class:
                typer.secho(f\"❌ Asset {self.asset} not configured in patch.\", fg=typer.colors.RED)
                typer.secho(f\"   Add it to asset_map in patch.py\", fg=typer.colors.RED)
                return

            interval_class = tf_map.get(self.tf.upper())
            if not interval_class:
                typer.secho(f\"❌ Timeframe {self.tf} not configured in patch.\", fg=typer.colors.RED)
                return

            # --- Date Logic ---
            end = datetime.now()
            start = end - timedelta(days=self.days)
            
            # Fix system time issue (if clock is set to future)
            if end > datetime.utcnow() + timedelta(days=1):
                end = datetime.utcnow()
                start = end - timedelta(days=self.days)

            # --- Fetch ---
            typer.echo(f\"   Range: {start.date()} to {end.date()}\")
            
            df = dukascopy_python.fetch(
                instrument=instrument_class,
                interval=interval_class,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=start,
                end=end,
            )
            
            # --- Formatting for Poet Parser ---
            df.reset_index(inplace=True)
            # Standardize columns to match parser expectation
            df.rename(columns={'Date': 'UTC', 'Datetime': 'UTC', 'timestamp': 'UTC'}, inplace=True)
            df = df[['UTC', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # --- Save ---
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            typer.secho(f\"✅ Saved {len(df)} bars to {save_path}\", fg=typer.colors.GREEN)
            
        except ImportError:
            typer.secho(f\"❌ 'dukascopy-python' library not found.\", fg=typer.colors.RED)
            typer.secho(f\"   Run: pip install dukascopy-python\", fg=typer.colors.YELLOW)
        except Exception as e:
            typer.secho(f\"❌ Dukascopy fetch failed: {e}\", fg=typer.colors.RED)
            raise e
"""

# ============================================
# 2. PATCHER LOGIC
# ============================================

def apply_patch():
    source_file = Path("poet_cli.py")
    
    if not source_file.exists():
        print(f" Error: {source_file} not found in current directory.")
        return

    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()

    print(f" Loading {source_file}...")

    # --- PATCH 1: Fix typer.echo -> typer.secho ---
    # Regex: Finds typer.echo(...) where 'fg=' is present, and changes it to secho
    old_echo_pattern = r'typer\.echo\((.*?)fg='
    new_echo = r'typer.secho(\1fg='
    content = re.sub(old_echo_pattern, new_echo, content)
    print("🔧 Fixed typer.echo color formatting errors.")

    # --- PATCH 2: Replace DataFetcher.fetch method ---
    # Finding the start of the fetch method
    start_marker = "    def fetch(self, save_path: Path):"
    if start_marker in content:
        print("🔧 Found existing fetch method. Replacing with Dukascopy logic...")
        
        # Find the end of the method (next 'def ' at same indent)
        parts = content.split(start_marker)
        
        # parts[0] is everything before the method
        # parts[1] is the method body + rest of file
        
        # Find the next 'def ' in the rest
        body_and_rest = parts[1]
        next_def_idx = body_and_rest.find("    def ")
        
        if next_def_idx != -1:
            rest_of_file = body_and_rest[next_def_idx:]
            content = parts[0] + NEW_DATAFETCHER_CODE + rest_of_file
        else:
            # If fetch is the last method in class
            content = parts[0] + NEW_DATAFETCHER_CODE
    else:
        print("  Could not find 'def fetch' method. Skipping patcher.")
        return

    # Write back
    with open(source_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f" Patch applied successfully to {source_file}")
    print("   1. Fixed 'typer.echo' errors.")
    print("   2. Replaced Fetcher with Dukascopy.")

if __name__ == "__main__":
    apply_patch()
