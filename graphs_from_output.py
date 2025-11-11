import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Vytvoření složky pro grafy
OUTPUT_GRAPHS_DIR = './output_graphs'
os.makedirs(OUTPUT_GRAPHS_DIR, exist_ok=True)

# --- FUNKCE PRO NAČTENÍ DAT ---
def parse_results_file(filepath):
    """Přečte jeden results.txt a vrátí seznam slovníků s daty pro každý kloub."""
    data = []
    current_joint = None
    joint_data = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Zkusíme jiné kódování, pokud utf-8 selže
        with open(filepath, 'r', encoding='cp1250') as f:
             lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            current_joint = line[:-1] # Odstraníme dvojtečku
            joint_data = {'joint': current_joint}
        elif ':' in line and current_joint:
            key, value = line.split(':', 1)
            key = key.strip()
            # Odstraníme symbol stupně a převedeme na číslo
            value_clean = value.strip().replace('°', '').replace(',', '.')

            try:
                if 'Minimální úhel' in key:
                    joint_data['min'] = float(value_clean)
                elif 'Maximální úhel' in key:
                    joint_data['max'] = float(value_clean)
                elif 'Průměrný úhel' in key:
                    joint_data['avg'] = float(value_clean)
                elif 'Počet platných měření' in key:
                    joint_data['count'] = int(value_clean)
                    # Máme vše potřebné pro tento kloub, uložíme a resetujeme
                    if 'min' in joint_data and 'max' in joint_data:
                        joint_data['rom'] = joint_data['max'] - joint_data['min']
                        data.append(joint_data.copy())
                    current_joint = None
            except ValueError:
                continue # Přeskočíme řádky, kde se nepovedl převod na číslo

    return data

def load_all_data(root_dir):
    """Projde celou strukturu složek a načte všechna data do DataFrame."""
    all_records = []
    # Procházíme složky. Předpokládáme strukturu: root / model / view / condition / results.txt
    for root, dirs, files in os.walk(root_dir):
        if 'results.txt' in files:
            # Získáme cestu a rozdělíme ji na části
            path_parts = os.path.normpath(root).split(os.sep)
            # Jednoduchá heuristika: předpokládáme, že poslední 3 složky jsou condition/view/model
            if len(path_parts) >= 3:
                condition = path_parts[-1]
                view = path_parts[-2]
                model = path_parts[-3]

                full_path = os.path.join(root, 'results.txt')
                file_data = parse_results_file(full_path)

                for record in file_data:
                    record['model'] = model
                    record['view'] = view
                    record['condition'] = condition
                    all_records.append(record)

    return pd.DataFrame(all_records)

# --- FUNKCE PRO VIZUALIZACI ---
def plot_model_comparison(df, model_name, view_type, joints_to_keep):
    """Vykreslí graf porovnání podmínek pro jeden model a pohled."""
    # Filtrace dat
    subset = df[(df['model'] == model_name) &
                (df['view'] == view_type) &
                (df['joint'].isin(joints_to_keep))].copy()

    if subset.empty:
        print(f"Žádná data pro graf: {model_name} - {view_type}")
        return

    # Nastavení pořadí pro osu X
    condition_order = ['minustwenty', 'minusten', 'zero', 'plusten', 'plustwenty']
    # Ujistíme se, že používáme jen ty podmínky, které skutečně máme v datech
    available_conditions = [c for c in condition_order if c in subset['condition'].unique()]
    subset['condition'] = pd.Categorical(subset['condition'], categories=available_conditions, ordered=True)
    subset = subset.sort_values(['joint', 'condition'])

    # Vytvoření grafu
    plt.figure(figsize=(14, 7))
    # Barplot zobrazí ROM
    ax = sns.barplot(data=subset, x='condition', y='rom', hue='joint', palette='viridis')

    # Přidání textových popisů (počet framů) do grafu
    # Procházíme sloupce a přidáváme text. Je to trochu trik, protože seaborn neposkytuje snadný přístup.
    for i, row in subset.iterrows():
        # Najdeme správnou pozici X pro daný sloupec (to je složité automatizovat dokonale pro seskupené bary)
        # Zjednodušení: vypíšeme hodnoty do konzole nebo použijeme interaktivní tooltips v jiném nástroji.
        # Zde zkusíme základní anotaci nad přibližnou pozicí.
        pass

    plt.title(f'Model: {model_name} | Pohled: {view_type.upper()} | ROM (Rozsah pohybu)')
    plt.ylabel('Rozsah pohybu (stupně)')
    plt.xlabel('Podmínka')
    plt.xticks(rotation=45)
    plt.legend(title='Kloub')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Uložení grafu
    filename = f"{model_name}_{view_type}_rom.png"
    filepath = os.path.join(OUTPUT_GRAPHS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graf uložen: {filename}")
    plt.close()

def plot_zero_targets(df, view, target_joints, target_values):
    """Vykreslí porovnání všech modelů v podmínce ZERO vůči referenčním hodnotám."""
    subset = df[(df['view'] == view) & (df['condition'] == 'zero')].copy()
    subset = subset[subset['joint'].isin(target_joints)]

    if subset.empty: return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=subset, x='model', y='rom', hue='joint', palette='deep')

    # Přidání referenčních čar
    colors = ['red', 'green', 'blue', 'orange']
    for i, (joint, target) in enumerate(target_values.items()):
        plt.axhline(y=target, color=colors[i % len(colors)], linestyle='--',
                    label=f'Cíl {joint} ({target}°)')

    plt.title(f'Porovnání modelů - {view.upper()} / ZERO vs. Cílové hodnoty')
    plt.ylabel('Rozsah pohybu (ROM)')
    plt.xlabel('Model')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Uložení grafu
    joints_str = '_'.join([j.replace(' ', '_') for j in target_joints])
    filename = f"zero_comparison_{view}_{joints_str}.png"
    filepath = os.path.join(OUTPUT_GRAPHS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graf uložen: {filename}")
    plt.close()

def calculate_deviations_from_zero(df):
    """Vypočítá odchylky jednotlivých podmínek od ZERO pro každý model a kloub."""
    deviations = []
    
    # Definice důležitých kloubů podle pohledu
    important_joints = {
        'front': ['Pravý loket', 'Levý loket', 'Pravé rameno', 'Levé rameno'],
        'side': ['Pravé koleno']
    }
    
    # Procházíme každý model, view a kloub
    for model in df['model'].unique():
        for view in df['view'].unique():
            # Filtrujeme jen důležité klouby pro daný pohled
            joints_to_process = important_joints.get(view, [])
            
            for joint in joints_to_process:
                # Získáme data pro tento model/view/joint
                subset = df[(df['model'] == model) & 
                           (df['view'] == view) & 
                           (df['joint'] == joint)]
                
                # Najdeme hodnotu ZERO
                zero_data = subset[subset['condition'] == 'zero']
                if zero_data.empty:
                    continue
                    
                zero_rom = zero_data['rom'].values[0]
                zero_avg = zero_data['avg'].values[0]
                
                # Porovnáme s ostatními podmínkami
                for condition in ['minustwenty', 'minusten', 'plusten', 'plustwenty']:
                    cond_data = subset[subset['condition'] == condition]
                    if cond_data.empty:
                        continue
                    
                    cond_rom = cond_data['rom'].values[0]
                    cond_avg = cond_data['avg'].values[0]
                    
                    # Vypočítáme odchylky
                    rom_deviation = cond_rom - zero_rom
                    avg_deviation = cond_avg - zero_avg
                    rom_deviation_percent = (rom_deviation / zero_rom * 100) if zero_rom != 0 else 0
                    
                    deviations.append({
                        'model': model,
                        'view': view,
                        'joint': joint,
                        'condition': condition,
                        'zero_rom': zero_rom,
                        'condition_rom': cond_rom,
                        'rom_deviation': rom_deviation,
                        'rom_deviation_percent': rom_deviation_percent,
                        'zero_avg': zero_avg,
                        'condition_avg': cond_avg,
                        'avg_deviation': avg_deviation
                    })
    
    return pd.DataFrame(deviations)

def calculate_model_average_deviations(deviations_df):
    """Vypočítá průměrnou odchylku pro každý model a každou podmínku zvlášť."""
    if deviations_df.empty:
        return pd.DataFrame()
    
    # Průměr absolutních odchylek pro každý model A PODMÍNKU
    model_summary = deviations_df.groupby(['model', 'condition']).agg({
        'rom_deviation': lambda x: abs(x).mean(),
        'rom_deviation_percent': lambda x: abs(x).mean(),
        'avg_deviation': lambda x: abs(x).mean()
    }).reset_index()
    
    model_summary.columns = ['model', 'condition', 'avg_abs_rom_deviation', 
                              'avg_abs_rom_deviation_percent', 'avg_abs_avg_deviation']
    
    return model_summary

def save_deviation_analysis(deviations_df, model_summary, output_dir):
    """Uloží analýzu odchylek do souborů."""
    # 1. Detailní odchylky
    detail_file = os.path.join(output_dir, 'deviations_detail.csv')
    deviations_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
    print(f"   💾 Detailní odchylky: deviations_detail.csv")
    
    # 2. Souhrn po modelech A PODMÍNKÁCH
    summary_file = os.path.join(output_dir, 'deviations_summary.csv')
    model_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"   💾 Souhrn odchylek: deviations_summary.csv")
    
    # 3. Textový report
    txt_file = os.path.join(output_dir, 'deviations_report.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANALÝZA ODCHYLEK OD PODMÍNKY ZERO\n")
        f.write("="*80 + "\n\n")
        
        condition_order = ['minustwenty', 'minusten', 'plusten', 'plustwenty']
        
        for condition in condition_order:
            f.write(f"\n{'='*80}\n")
            f.write(f"PODMÍNKA: {condition.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            cond_summary = model_summary[model_summary['condition'] == condition].sort_values('avg_abs_rom_deviation')
            
            if not cond_summary.empty:
                f.write(f"{'Model':<20} {'ROM odchylka (°)':<20} {'ROM odchylka (%)':<20} {'AVG odchylka (°)':<20}\n")
                f.write("-"*80 + "\n")
                
                for _, row in cond_summary.iterrows():
                    f.write(f"{row['model']:<20} {row['avg_abs_rom_deviation']:<20.2f} "
                           f"{row['avg_abs_rom_deviation_percent']:<20.2f} {row['avg_abs_avg_deviation']:<20.2f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailní rozpis po modelech
        f.write("DETAILNÍ ROZPIS PO MODELECH:\n")
        f.write("="*80 + "\n\n")
        
        for model in deviations_df['model'].unique():
            model_data = deviations_df[deviations_df['model'] == model]
            f.write(f"\n{model.upper()}\n")
            f.write("-"*80 + "\n")
            
            for condition in condition_order:
                cond_data = model_data[model_data['condition'] == condition]
                if cond_data.empty:
                    continue
                    
                f.write(f"\n  Podmínka: {condition}\n")
                
                for view in cond_data['view'].unique():
                    view_data = cond_data[cond_data['view'] == view]
                    f.write(f"\n    Pohled: {view}\n")
                    
                    for _, row in view_data.iterrows():
                        f.write(f"      {row['joint']:<20} ZERO: {row['zero_rom']:>6.2f}°  "
                               f"→ {row['condition_rom']:>6.2f}°  "
                               f"Odchylka: {row['rom_deviation']:>+7.2f}° ({row['rom_deviation_percent']:>+6.2f}%)\n")
            
            f.write("\n")
    
    print(f"   💾 Textový report: deviations_report.txt")

def plot_model_deviations(model_summary, output_dir):
    """Vykreslí grafy průměrných odchylek modelů od ZERO - samostatně pro každou podmínku."""
    if model_summary.empty:
        return
    
    condition_order = ['minustwenty', 'minusten', 'plusten', 'plustwenty']
    
    for condition in condition_order:
        cond_data = model_summary[model_summary['condition'] == condition].sort_values('avg_abs_rom_deviation')
        
        if cond_data.empty:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Průměrné odchylky od ZERO - Podmínka: {condition.upper()}', 
                     fontsize=14, fontweight='bold')
        
        # Graf 1: Absolutní odchylka ROM
        ax1 = axes[0]
        bars1 = ax1.barh(cond_data['model'], cond_data['avg_abs_rom_deviation'], 
                         color='steelblue', alpha=0.8)
        ax1.set_xlabel('Průměrná absolutní odchylka ROM (°)')
        ax1.set_ylabel('Model')
        ax1.set_title('Průměrná odchylka - ROM (stupně)')
        ax1.grid(axis='x', alpha=0.3)
        
        # Přidání hodnot na grafy
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}°', ha='left', va='center', fontsize=9)
        
        # Graf 2: Procentuální odchylka ROM
        ax2 = axes[1]
        bars2 = ax2.barh(cond_data['model'], cond_data['avg_abs_rom_deviation_percent'], 
                         color='coral', alpha=0.8)
        ax2.set_xlabel('Průměrná absolutní odchylka ROM (%)')
        ax2.set_ylabel('Model')
        ax2.set_title('Průměrná odchylka - ROM (procenta)')
        ax2.grid(axis='x', alpha=0.3)
        
        # Přidání hodnot na grafy
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        filename = f'model_deviations_{condition}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   ✅ Graf uložen: {filename}")
        plt.close()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graf uložen: {filename}")
    plt.close()

def plot_condition_deviations(deviations_df, output_dir):
    """Vykreslí graf odchylek pro každou podmínku napříč modely."""
    if deviations_df.empty:
        return
    
    # Průměrná odchylka pro každou podmínku
    condition_summary = deviations_df.groupby('condition').agg({
        'rom_deviation': lambda x: abs(x).mean()
    }).reset_index()
    
    condition_order = ['minustwenty', 'minusten', 'plusten', 'plustwenty']
    condition_summary['condition'] = pd.Categorical(
        condition_summary['condition'], 
        categories=condition_order, 
        ordered=True
    )
    condition_summary = condition_summary.sort_values('condition')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(condition_summary['condition'], 
                   condition_summary['rom_deviation'],
                   color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    
    plt.xlabel('Podmínka')
    plt.ylabel('Průměrná absolutní odchylka ROM (°)')
    plt.title('Průměrná odchylka od ZERO podle podmínky (napříč všemi modely)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Přidání hodnot
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}°', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    filename = 'condition_deviations_from_zero.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graf uložen: {filename}")
    plt.close()

def plot_detailed_model_deviations(deviations_df, model_summary, output_dir):
    """
    Vykreslí detailní graf pro každý model zobrazující odchylky jednotlivých podmínek
    s porovnáním průměrů podmínek z model_summary.
    """
    if deviations_df.empty or model_summary.empty:
        return
    
    condition_order = ['minustwenty', 'minusten', 'plusten', 'plustwenty']
    models = sorted(deviations_df['model'].unique())
    
    # Pro každý model vytvoříme subplot
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Data pro tento model z model_summary (obsahuje průměry po podmínkách)
        model_summary_data = model_summary[model_summary['model'] == model]
        
        if model_summary_data.empty:
            ax.set_visible(False)
            continue
        
        # Příprava dat pro graf
        conditions = []
        means = []
        
        for condition in condition_order:
            cond_summary = model_summary_data[model_summary_data['condition'] == condition]
            if not cond_summary.empty:
                conditions.append(condition)
                means.append(cond_summary['avg_abs_rom_deviation'].values[0])
        
        if not means:
            ax.set_visible(False)
            continue
        
        # Vytvoření bar grafu
        colors = ['#e74c3c', '#e67e22', '#3498db', '#9b59b6']  # červená, oranžová, modrá, fialová
        bars = ax.bar(conditions, means, alpha=0.8, 
                      color=[colors[condition_order.index(c)] for c in conditions],
                      edgecolor='black', linewidth=1.5)
        
        # Celkový průměr pro tento model (průměr přes všechny podmínky)
        overall_avg = sum(means) / len(means)
        ax.axhline(y=overall_avg, color='darkgreen', linestyle='--', linewidth=2, 
                   label=f'Celkový průměr: {overall_avg:.2f}°', alpha=0.7)
        
        # Označení hodnot nad sloupci
        for i, (bar, mean, condition) in enumerate(zip(bars, means, conditions)):
            height = bar.get_height()
            # Vypočítáme rozdíl od celkového průměru
            diff = mean - overall_avg
            text_color = 'darkgreen' if abs(diff) < overall_avg * 0.15 else 'darkred'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}°\n({diff:+.2f}°)',
                   ha='center', va='bottom', fontsize=9, color=text_color, fontweight='bold')
        
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Podmínka', fontsize=10)
        ax.set_ylabel('Průměrná abs. odchylka ROM (°)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Skrytí nepoužitých subplotů
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Detailní odchylky od ZERO pro jednotlivé modely a podmínky', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = 'detailed_model_condition_deviations.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graf uložen: {filename}")
    plt.close()

# ==========================================
# HLAVNÍ ČÁST SKRIPTU
# ==========================================

# 1. NASTAV CESTU K DATŮM
root_directory = r'./output' # <--- SEM ZADEJ SVOU CESTU

print(f"📁 Složka pro grafy: {OUTPUT_GRAPHS_DIR}")

# 2. Načtení dat (pokud složka existuje)
if os.path.exists(root_directory):
    print("Načítám data...")
    df = load_all_data(root_directory)

    if not df.empty:
        # Uložení pro kontrolu
        df.to_csv(os.path.join(root_directory, 'vysledna_analyza.csv'), index=False, encoding='utf-8-sig')
        print("Data uložena do 'vysledna_analyza.csv'")

        # 3. Vykreslení grafů
        print("Generuji grafy...")
        models = df['model'].unique()
        all_joints = df['joint'].unique()

        for model in models:
            # a) Graf pro SIDE (jen pravé koleno a kyčel)
            plot_model_comparison(df, model, 'side', ['Pravé koleno', 'Pravá kyčel'])

            # b) Graf pro FRONT (vše kromě kolen a kyčlí)
            front_joints = [j for j in all_joints if not any(x in j.lower() for x in ['koleno', 'kyčel'])]
            plot_model_comparison(df, model, 'front', front_joints)

        # 4. Porovnávací grafy ZERO napříč modely
        # Side cíle: Pravé koleno 125, Levé koleno 127
        plot_zero_targets(df, 'side',
                         ['Pravé koleno', 'Levé koleno'],
                         {'Pravé koleno': 125, 'Levé koleno': 127})

        # Front cíle: Pravý loket 135, Levý loket 134
        plot_zero_targets(df, 'front',
                         ['Pravý loket', 'Levý loket'],
                         {'Pravý loket': 135, 'Levý loket': 134})

        # 5. ANALÝZA ODCHYLEK OD ZERO
        print("\n" + "="*60)
        print("📊 Analýza odchylek od podmínky ZERO...")
        print("="*60)
        
        deviations_df = calculate_deviations_from_zero(df)
        if not deviations_df.empty:
            model_summary = calculate_model_average_deviations(deviations_df)
            
            # Uložení do souborů
            save_deviation_analysis(deviations_df, model_summary, OUTPUT_GRAPHS_DIR)
            
            # Grafy odchylek
            plot_model_deviations(model_summary, OUTPUT_GRAPHS_DIR)
            plot_condition_deviations(deviations_df, OUTPUT_GRAPHS_DIR)
            plot_detailed_model_deviations(deviations_df, model_summary, OUTPUT_GRAPHS_DIR)
            
            print("\n✅ Analýza odchylek dokončena!")
        else:
            print("⚠️ Žádná data pro analýzu odchylek")

        print("\nHotovo.")
    else:
        print("Nenalezena žádná data (žádné results.txt soubory).")
else:
    print(f"Cesta neexistuje: {root_directory}")