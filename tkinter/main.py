import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import threading
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class PoseAnalyzer:
    def __init__(self):
        # Inicializace MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Slovník pro uchování měření
        self.measurements = []
        self.current_frame = None
        self.video_path = None
        self.is_recording = False
        
    def calculate_angle(self, point1, point2, point3):
        """Vypočítá úhel mezi třemi body"""
        # Vytvoření vektorů
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Výpočet úhlu
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def extract_key_angles(self, landmarks):
        """Extrahuje klíčové úhly z detekovaných bodů"""
        if not landmarks:
            return None
            
        # Převod landmarks na body
        points = {}
        for i, landmark in enumerate(landmarks.landmark):
            points[i] = [landmark.x, landmark.y]
        
        angles = {}
        
        try:
            # Úhel levého lokte (rameno-loket-zápěstí)
            if all(i in points for i in [11, 13, 15]):
                angles['left_elbow'] = self.calculate_angle(points[11], points[13], points[15])
            
            # Úhel pravého lokte
            if all(i in points for i in [12, 14, 16]):
                angles['right_elbow'] = self.calculate_angle(points[12], points[14], points[16])
            
            # Úhel levého kolena (kyčel-koleno-kotník)
            if all(i in points for i in [23, 25, 27]):
                angles['left_knee'] = self.calculate_angle(points[23], points[25], points[27])
            
            # Úhel pravého kolena
            if all(i in points for i in [24, 26, 28]):
                angles['right_knee'] = self.calculate_angle(points[24], points[26], points[28])
            
            # Úhel levého ramena (trup-rameno-loket)
            if all(i in points for i in [23, 11, 13]):
                angles['left_shoulder'] = self.calculate_angle(points[23], points[11], points[13])
            
            # Úhel pravého ramena
            if all(i in points for i in [24, 12, 14]):
                angles['right_shoulder'] = self.calculate_angle(points[24], points[12], points[14])
            
            #Úhel levé kyčle (trup-kyčel-koleno)
            if all(i in points for i in [11, 23, 25]):
                angles['left_hip'] = self.calculate_angle(points[11], points[23], points[25])
            
            # Úhel pravé kyčle
            if all(i in points for i in [12, 24, 26]):
                angles['right_hip'] = self.calculate_angle(points[12], points[24], points[26])
                
        except Exception as e:
            print(f"Chyba při výpočtu úhlů: {e}")
            
        return angles
    
    def process_frame(self, frame):
        """Zpracuje jeden frame videa"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        # Kreslení skeletu
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Extrakce úhlů
            angles = self.extract_key_angles(results.pose_landmarks)
            if angles:
                # Přidání časového razítka
                measurement = {
                    'timestamp': datetime.now().isoformat(),
                    'angles': angles
                }
                if self.is_recording:
                    self.measurements.append(measurement)
                
                # Zobrazení úhlů na obraze - větší text pro lepší čitelnost
                y_offset = 40
                for joint, angle in angles.items():
                    text = f"{joint}: {angle:.1f}°"
                    cv2.putText(frame, text, (15, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 35
        
        return frame, results.pose_landmarks

class PoseAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikace pro měření rozsahu pohybu")
        self.root.geometry("1800x1200")  # Zvětšené okno pro 2,5K monitor
        
        # Konfigurace pro vysoké DPI
        self.root.tk.call('tk', 'scaling', 1.5)  # Zvětšení UI elementů
        
        # Konfigurace stylů pro větší text
        style = ttk.Style()
        style.configure('TLabel', font=('TkDefaultFont', 11))
        style.configure('TButton', font=('TkDefaultFont', 10))
        style.configure('TCheckbutton', font=('TkDefaultFont', 10))
        style.configure('TNotebook.Tab', font=('TkDefaultFont', 11))
        
        # Inicializace analyzátoru
        self.analyzer = PoseAnalyzer()
        self.video_capture = None
        self.is_playing = False
        self.current_video_frame = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Nastavení uživatelského rozhraní"""
        # Hlavní notebook pro tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab pro analýzu videa
        self.video_frame = ttk.Frame(notebook)
        notebook.add(self.video_frame, text="Analýza videa")
        self.setup_video_tab()
        
        # Tab pro výsledky
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.results_frame, text="Výsledky")
        self.setup_results_tab()
        
        # Tab pro export
        self.export_frame = ttk.Frame(notebook)
        notebook.add(self.export_frame, text="Export dat")
        self.setup_export_tab()
    
    def setup_video_tab(self):
        """Nastavení tabu pro analýzu videa"""
        # Ovládací panel
        control_frame = ttk.Frame(self.video_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Načíst video", 
                  command=self.load_video).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Spustit kameru", 
                  command=self.start_camera).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Přehrát/Pauza", 
                  command=self.toggle_playback).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Zastavit", 
                  command=self.stop_playback).pack(side='left', padx=5)
        
        # Checkbox pro nahrávání měření
        self.recording_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Nahrávat měření", 
                       variable=self.recording_var,
                       command=self.toggle_recording).pack(side='left', padx=5)
        
        # Frame pro video
        self.video_display = ttk.Label(self.video_frame, text="Vyberte video nebo spusťte kameru", 
                                      font=('TkDefaultFont', 12))
        self.video_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Připraven")
        status_bar = ttk.Label(self.video_frame, textvariable=self.status_var)
        status_bar.pack(fill='x', padx=5, pady=5)
    
    def setup_results_tab(self):
        """Nastavení tabu pro výsledky"""
        # Frame pro grafy
        self.graph_frame = ttk.Frame(self.results_frame)
        self.graph_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Ovládání grafů
        graph_control = ttk.Frame(self.results_frame)
        graph_control.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(graph_control, text="Zobrazit grafy úhlů", 
                  command=self.plot_angles).pack(side='left', padx=5)
        ttk.Button(graph_control, text="Statistiky", 
                  command=self.show_statistics).pack(side='left', padx=5)
        ttk.Button(graph_control, text="Vyčistit data", 
                  command=self.clear_measurements).pack(side='left', padx=5)
        
        # Text area pro statistiky
        self.stats_text = scrolledtext.ScrolledText(self.results_frame, height=12, 
                                                   font=('TkDefaultFont', 11))
        self.stats_text.pack(fill='x', padx=10, pady=10)
    
    def setup_export_tab(self):
        """Nastavení tabu pro export"""
        export_control = ttk.Frame(self.export_frame)
        export_control.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(export_control, text="Export do JSON", 
                  command=self.export_json).pack(side='left', padx=5)
        ttk.Button(export_control, text="Export do XML", 
                  command=self.export_xml).pack(side='left', padx=5)
        ttk.Button(export_control, text="Generovat report", 
                  command=self.generate_report).pack(side='left', padx=5)
        
        # Text preview
        self.export_preview = scrolledtext.ScrolledText(self.export_frame, 
                                                       font=('TkDefaultFont', 11))
        self.export_preview.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_video(self):
        """Načtení video souboru"""
        file_path = filedialog.askopenfilename(
            title="Vyberte video soubor",
            filetypes=[("Video soubory", "*.mp4 *.avi *.mov *.mkv"), ("Všechny soubory", "*.*")]
        )
        
        if file_path:
            self.analyzer.video_path = file_path
            self.video_capture = cv2.VideoCapture(file_path)
            self.status_var.set(f"Video načteno: {os.path.basename(file_path)}")
    
    def start_camera(self):
        """Spuštění kamery"""
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Chyba", "Nelze spustit kameru")
            return
        self.status_var.set("Kamera spuštěna")
    
    def toggle_playback(self):
        """Přepnutí přehrávání"""
        if not self.video_capture:
            messagebox.showwarning("Upozornění", "Nejprve načtěte video nebo spusťte kameru")
            return
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_video()
        self.status_var.set("Přehrávání" if self.is_playing else "Pozastaveno")
    
    def stop_playback(self):
        """Zastavení přehrávání"""
        self.is_playing = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.status_var.set("Zastaveno")
    
    def toggle_recording(self):
        """Přepnutí nahrávání měření"""
        self.analyzer.is_recording = self.recording_var.get()
        status = "Nahrávání zapnuto" if self.analyzer.is_recording else "Nahrávání vypnuto"
        self.status_var.set(status)
    
    def play_video(self):
        """Přehrávání videa s analýzou"""
        def video_thread():
            while self.is_playing and self.video_capture:
                ret, frame = self.video_capture.read()
                if not ret:
                    self.is_playing = False
                    break
                
                # Zpracování frame
                processed_frame, landmarks = self.analyzer.process_frame(frame)
                
                # Převod pro Tkinter - větší velikost pro 2,5K monitor
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                processed_frame = cv2.resize(processed_frame, (960, 720))  # Zvětšené video
                
                # Aktualizace GUI (musí být v main thread)
                self.root.after(0, self.update_video_display, processed_frame)
                
                # FPS kontrola
                cv2.waitKey(33)  # ~30 FPS
        
        if self.is_playing:
            thread = threading.Thread(target=video_thread)
            thread.daemon = True
            thread.start()
    
    def update_video_display(self, frame):
        """Aktualizace zobrazení videa"""
        # Převod numpy array na PhotoImage
        from PIL import Image, ImageTk
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        
        self.video_display.configure(image=photo)
        self.video_display.image = photo  # Uchování reference
    
    def plot_angles(self):
        """Zobrazení grafů úhlů"""
        if not self.analyzer.measurements:
            messagebox.showwarning("Upozornění", "Žádná data k zobrazení")
            return
        
        # Vyčištění předchozího grafu
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Příprava dat
        angles_data = defaultdict(list)
        timestamps = []
        
        for measurement in self.analyzer.measurements:
            timestamps.append(len(timestamps))  # Jednoduché indexování
            for angle_name, angle_value in measurement['angles'].items():
                angles_data[angle_name].append(angle_value)
        
        # Vytvoření grafu - větší velikost pro 2,5K monitor
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Vývoj úhlů v čase', fontsize=16)
        
        angle_names = list(angles_data.keys())[:4]  # Zobrazit prvních 4 úhly
        
        for i, angle_name in enumerate(angle_names):
            ax = axes[i // 2, i % 2]
            ax.plot(timestamps[:len(angles_data[angle_name])], angles_data[angle_name], linewidth=2)
            ax.set_title(f'{angle_name}', fontsize=14)
            ax.set_xlabel('Frame', fontsize=12)
            ax.set_ylabel('Úhel (stupně)', fontsize=12)
            ax.tick_params(labelsize=11)
            ax.grid(True)
        
        # Zobrazení v GUI
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_statistics(self):
        """Zobrazení statistik"""
        if not self.analyzer.measurements:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Žádná data k analýze")
            return
        
        # Výpočet statistik
        stats = {}
        all_angles = defaultdict(list)
        
        for measurement in self.analyzer.measurements:
            for angle_name, angle_value in measurement['angles'].items():
                all_angles[angle_name].append(angle_value)
        
        # Zobrazení statistik
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"STATISTIKY MĚŘENÍ\n")
        self.stats_text.insert(tk.END, f"================\n\n")
        self.stats_text.insert(tk.END, f"Celkem měření: {len(self.analyzer.measurements)}\n\n")
        
        for angle_name, values in all_angles.items():
            if values:
                mean_angle = np.mean(values)
                std_angle = np.std(values)
                min_angle = np.min(values)
                max_angle = np.max(values)
                range_angle = max_angle - min_angle
                
                self.stats_text.insert(tk.END, f"{angle_name.upper()}:\n")
                self.stats_text.insert(tk.END, f"  Průměr: {mean_angle:.2f}°\n")
                self.stats_text.insert(tk.END, f"  Směrodatná odchylka: {std_angle:.2f}°\n")
                self.stats_text.insert(tk.END, f"  Minimum: {min_angle:.2f}°\n")
                self.stats_text.insert(tk.END, f"  Maximum: {max_angle:.2f}°\n")
                self.stats_text.insert(tk.END, f"  Rozsah pohybu: {range_angle:.2f}°\n\n")
    
    def clear_measurements(self):
        """Vyčištění naměřených dat"""
        if messagebox.askyesno("Potvrzení", "Opravdu chcete vymazat všechna naměřená data?"):
            self.analyzer.measurements = []
            self.stats_text.delete(1.0, tk.END)
            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            self.status_var.set("Data vymazána")
    
    def export_json(self):
        """Export dat do JSON"""
        if not self.analyzer.measurements:
            messagebox.showwarning("Upozornění", "Žádná data k exportu")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON soubory", "*.json"), ("Všechny soubory", "*.*")]
        )
        
        if file_path:
            data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'total_measurements': len(self.analyzer.measurements),
                    'video_source': self.analyzer.video_path or 'Kamera'
                },
                'measurements': self.analyzer.measurements
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Export", f"Data exportována do: {file_path}")
            
            # Preview
            self.export_preview.delete(1.0, tk.END)
            self.export_preview.insert(tk.END, json.dumps(data, indent=2, ensure_ascii=False)[:2000] + "...")
    
    def export_xml(self):
        """Export dat do XML"""
        if not self.analyzer.measurements:
            messagebox.showwarning("Upozornění", "Žádná data k exportu")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xml",
            filetypes=[("XML soubory", "*.xml"), ("Všechny soubory", "*.*")]
        )
        
        if file_path:
            root = ET.Element("PoseAnalysis")
            
            # Metadata
            metadata = ET.SubElement(root, "Metadata")
            ET.SubElement(metadata, "ExportDate").text = datetime.now().isoformat()
            ET.SubElement(metadata, "TotalMeasurements").text = str(len(self.analyzer.measurements))
            ET.SubElement(metadata, "VideoSource").text = self.analyzer.video_path or "Kamera"
            
            # Measurements
            measurements = ET.SubElement(root, "Measurements")
            
            for i, measurement in enumerate(self.analyzer.measurements):
                measurement_elem = ET.SubElement(measurements, "Measurement", id=str(i))
                ET.SubElement(measurement_elem, "Timestamp").text = measurement['timestamp']
                
                angles_elem = ET.SubElement(measurement_elem, "Angles")
                for angle_name, angle_value in measurement['angles'].items():
                    angle_elem = ET.SubElement(angles_elem, "Angle", name=angle_name)
                    angle_elem.text = str(angle_value)
            
            tree = ET.ElementTree(root)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)
            
            messagebox.showinfo("Export", f"Data exportována do: {file_path}")
            
            # Preview
            self.export_preview.delete(1.0, tk.END)
            xml_str = ET.tostring(root, encoding='unicode')[:2000]
            self.export_preview.insert(tk.END, xml_str + "...")
    
    def generate_report(self):
        """Generování textového reportu"""
        if not self.analyzer.measurements:
            messagebox.showwarning("Upozornění", "Žádná data pro report")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Textové soubory", "*.txt"), ("Všechny soubory", "*.*")]
        )
        
        if file_path:
            # Výpočet statistik
            all_angles = defaultdict(list)
            for measurement in self.analyzer.measurements:
                for angle_name, angle_value in measurement['angles'].items():
                    all_angles[angle_name].append(angle_value)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("REPORT ANALÝZY ROZSAHU POHYBU\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Datum analýzy: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
                f.write(f"Zdroj videa: {self.analyzer.video_path or 'Kamera'}\n")
                f.write(f"Celkem měření: {len(self.analyzer.measurements)}\n\n")
                
                f.write("STATISTIKY JEDNOTLIVÝCH KLOUBŮ:\n")
                f.write("-" * 40 + "\n\n")
                
                for angle_name, values in all_angles.items():
                    if values:
                        mean_angle = np.mean(values)
                        std_angle = np.std(values)
                        min_angle = np.min(values)
                        max_angle = np.max(values)
                        range_angle = max_angle - min_angle
                        
                        f.write(f"{angle_name.upper().replace('_', ' ')}:\n")
                        f.write(f"  Průměrný úhel: {mean_angle:.2f}°\n")
                        f.write(f"  Směrodatná odchylka: {std_angle:.2f}°\n")
                        f.write(f"  Minimální úhel: {min_angle:.2f}°\n")
                        f.write(f"  Maximální úhel: {max_angle:.2f}°\n")
                        f.write(f"  Rozsah pohybu: {range_angle:.2f}°\n")
                        
                        # Hodnocení rozsahu pohybu
                        if 'elbow' in angle_name:
                            normal_range = "130-150°"
                            assessment = "Normální" if 130 <= range_angle <= 150 else "Mimo normu"
                        elif 'knee' in angle_name:
                            normal_range = "130-140°"
                            assessment = "Normální" if 130 <= range_angle <= 140 else "Mimo normu"
                        elif 'shoulder' in angle_name:
                            normal_range = "150-180°"
                            assessment = "Normální" if 150 <= range_angle <= 180 else "Mimo normu"
                        else:
                            normal_range = "N/A"
                            assessment = "N/A"
                        
                        f.write(f"  Normální rozsah: {normal_range}\n")
                        f.write(f"  Hodnocení: {assessment}\n\n")
                
                f.write("POZNÁMKY:\n")
                f.write("-" * 10 + "\n")
                f.write("- Toto měření je pouze orientační a nenahrazuje odborné vyšetření\n")
                f.write("- Pro přesné diagnostické účely konzultujte s fyzioterapeutem\n")
                f.write("- Normální rozsahy jsou orientační a mohou se lišit mezi jednotlivci\n")
            
            messagebox.showinfo("Report", f"Report vygenerován: {file_path}")
            
            # Preview
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()[:2000]
                self.export_preview.delete(1.0, tk.END)
                self.export_preview.insert(tk.END, content + "...")

def main():
    # Kontrola dostupnosti knihoven
    try:
        import mediapipe
        import cv2
        import matplotlib.pyplot as plt
        from PIL import Image, ImageTk
    except ImportError as e:
        print(f"Chybí požadovaná knihovna: {e}")
        print("Nainstalujte požadované knihovny:")
        print("pip install mediapipe opencv-python matplotlib pillow")
        return
    
    root = tk.Tk()
    app = PoseAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()