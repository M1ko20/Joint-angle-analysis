"""
Utility modul pro export dat v různých formátech
"""
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

class DataExporter:
    """Třída pro export analýz v různých formátech"""
    
    @staticmethod
    def export_json(data, filepath):
        """Exportuje data do JSON"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'video_path': data.get('video_path'),
            'model': data.get('model'),
            'total_frames': data.get('total_frames'),
            'selected_joints': data.get('selected_joints'),
            'statistics': data.get('statistics'),
            'angles_summary': {
                joint: {
                    'values': angles,
                    'count': len(angles)
                }
                for joint, angles in data.get('angles', {}).items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_xml(data, filepath):
        """Exportuje data do XML"""
        root = ET.Element('analysis')
        
        # Metadata
        metadata = ET.SubElement(root, 'metadata')
        ET.SubElement(metadata, 'timestamp').text = datetime.now().isoformat()
        ET.SubElement(metadata, 'video').text = str(data.get('video_path', ''))
        ET.SubElement(metadata, 'model').text = str(data.get('model', ''))
        ET.SubElement(metadata, 'total_frames').text = str(data.get('total_frames', 0))
        
        # Vybrané klouby
        joints_elem = ET.SubElement(root, 'selected_joints')
        for joint in data.get('selected_joints', []):
            ET.SubElement(joints_elem, 'joint').text = joint
        
        # Statistika
        if data.get('statistics'):
            stats_root = ET.SubElement(root, 'statistics')
            for joint, stats in data['statistics'].items():
                joint_elem = ET.SubElement(stats_root, 'joint', name=joint)
                ET.SubElement(joint_elem, 'average').text = f"{stats.get('average', 0):.2f}"
                ET.SubElement(joint_elem, 'minimum').text = f"{stats.get('min', 0):.2f}"
                ET.SubElement(joint_elem, 'maximum').text = f"{stats.get('max', 0):.2f}"
                ET.SubElement(joint_elem, 'std_dev').text = f"{stats.get('std_dev', 0):.2f}"
                ET.SubElement(joint_elem, 'count').text = str(stats.get('count', 0))
        
        # Úhly (zkrácená verze - jen počet)
        if data.get('angles'):
            angles_root = ET.SubElement(root, 'angles_summary')
            for joint, angles in data['angles'].items():
                angle_elem = ET.SubElement(angles_root, 'joint', name=joint)
                ET.SubElement(angle_elem, 'count').text = str(len(angles))
                if angles:
                    ET.SubElement(angle_elem, 'first').text = f"{angles[0]:.2f}"
                    ET.SubElement(angle_elem, 'last').text = f"{angles[-1]:.2f}"
        
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
    
    @staticmethod
    def export_csv(data, filepath):
        """Exportuje data do CSV"""
        stats = data.get('statistics', {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Hlavička
            writer.writerow(['Kloub', 'Průměr', 'Minimum', 'Maximum', 'Std Dev', 'Počet'])
            
            # Data
            for joint, stat in stats.items():
                writer.writerow([
                    joint,
                    f"{stat.get('average', 0):.2f}",
                    f"{stat.get('min', 0):.2f}",
                    f"{stat.get('max', 0):.2f}",
                    f"{stat.get('std_dev', 0):.2f}",
                    stat.get('count', 0)
                ])
    
    @staticmethod
    def export_html_report(data, filepath):
        """Exportuje data jako HTML report"""
        html = """<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analýza pohybu - Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 20px;
        }
        .metadata {
            background-color: #f9f9f9;
            padding: 10px;
            border-left: 4px solid #007bff;
            margin: 10px 0;
        }
        .metadata p {
            margin: 5px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .timestamp {
            color: #999;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Analýza Pohybu - Detailní Report</h1>
        <p class="timestamp">Vygenerováno: {timestamp}</p>
        
        <div class="metadata">
            <h2>Informace o analýze</h2>
            <p><strong>Video:</strong> {video_path}</p>
            <p><strong>Model:</strong> {model}</p>
            <p><strong>Celkem snímků:</strong> {total_frames}</p>
        </div>
        
        <h2>Statistika úhlů</h2>
        <table>
            <thead>
                <tr>
                    <th>Kloub</th>
                    <th>Průměr</th>
                    <th>Minimum</th>
                    <th>Maximum</th>
                    <th>Std Dev</th>
                    <th>Počet měření</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        
        stats = data.get('statistics', {})
        table_rows = ""
        
        for joint, stat in stats.items():
            table_rows += f"""
                <tr>
                    <td>{joint}</td>
                    <td>{stat.get('average', 0):.2f}°</td>
                    <td>{stat.get('min', 0):.2f}°</td>
                    <td>{stat.get('max', 0):.2f}°</td>
                    <td>{stat.get('std_dev', 0):.2f}</td>
                    <td>{stat.get('count', 0)}</td>
                </tr>
            """
        
        html = html.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            video_path=str(data.get('video_path', '')),
            model=str(data.get('model', '')),
            total_frames=data.get('total_frames', 0),
            table_rows=table_rows
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
