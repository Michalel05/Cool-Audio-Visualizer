import sys
from collections import deque
import numpy as np
import pyaudiowpatch as pyaudio
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# --- Konfiguration ---
CHUNK = 512             # Audio-Einlesegröße (Latenz ~10-11ms)
FFT_SIZE = 2048         # FFT-Fenstergröße (Erhöht Bass-Auflösung auf ~23 Hz pro Bin)
NUM_BARS = 60           # Anzahl der visuellen Balken

class AudioVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 1. PyAudio & WASAPI Loopback ---
        self.p = pyaudio.PyAudio()
        try:
            self.wasapi_info = self.p.get_default_wasapi_loopback()
        except OSError:
            print("Fehler: Kein WASAPI-Loopback-Gerät gefunden!")
            sys.exit(1)

        self.sample_rate = int(self.wasapi_info['defaultSampleRate'])
        self.num_channels = self.wasapi_info['maxInputChannels']

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.num_channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.wasapi_info['index'],
            frames_per_buffer=CHUNK
        )

        # Ringpuffer für hochauflösende FFT ohne Latenzverlust
        self.audio_buffer = deque(maxlen=FFT_SIZE)
        self.audio_buffer.extend(np.zeros(FFT_SIZE))

        # Hann-Fenster für die FFT_SIZE
        self.window = np.hanning(FFT_SIZE)

        # --- 2. Logarithmische Frequenz-Einteilung & Frequenz-Tilt ---
        min_freq = 30
        max_freq = 15000
        
        # Frequenzpunkte für die Bins
        freq_points = np.logspace(np.log10(min_freq), np.log10(max_freq), NUM_BARS + 1)
        fft_freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / self.sample_rate)
        
        self.bin_indices = np.digitize(freq_points, fft_freqs) - 1
        self.bin_indices = np.clip(self.bin_indices, 0, len(fft_freqs) - 1)

        # Equalization-Kurve: Bass leicht dämpfen, Höhen/Mitten anheben (+dB/Oktave)
        # Verhindert, dass der Bass alle anderen Balken überrollt.
        raw_freqs = np.geomspace(min_freq, max_freq, NUM_BARS)
        self.eq_weights = np.power(raw_freqs / min_freq, 0.75) * 0.2

        self.prev_heights = np.zeros(NUM_BARS)

        # --- 3. PyQtGraph GUI ---
        self.setWindowTitle("Megageiler AV")
        self.resize(1000, 500)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground('#0d0f17')
        layout.addWidget(self.graph_widget)

        self.plot = self.graph_widget.addPlot()
        self.plot.setYRange(0, 120)
        self.plot.setXRange(-0.5, NUM_BARS - 0.5)
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        self.plot.setMouseEnabled(x=False, y=False)

        # Farbverlauf-Gradient erzeugen (Cyan -> Violett -> Magenta)
        colors = []
        for i in range(NUM_BARS):
            r = int(120 + 135 * (i / NUM_BARS))
            g = int(80 * (1 - i / NUM_BARS))
            b = int(240 - 40 * (i / NUM_BARS))
            colors.append((r, g, b))

        x_pos = np.arange(NUM_BARS)
        self.bars = pg.BarGraphItem(
            x=x_pos,
            height=np.zeros(NUM_BARS),
            width=0.75,
            brushes=[pg.mkBrush(c) for c in colors],
            pen=pg.mkPen(None)
        )
        self.plot.addItem(self.bars)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualizer)
        self.timer.start(16)

    #Vollbild eines Tages hier

    def update_visualizer(self):
        try:
            if self.stream.get_read_available() < CHUNK:
                return

            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            if self.num_channels > 1:
                audio_data = audio_data.reshape(-1, self.num_channels).mean(axis=1)

            # Neue Daten in den Puffer schieben
            self.audio_buffer.extend(audio_data)
            buffer_np = np.array(self.audio_buffer)

            # FFT über 2048 Samples berechnen
            windowed_data = buffer_np * self.window
            fft_data = np.abs(np.fft.rfft(windowed_data))

            # Frequenzen in Bins zusammenfassen (Averaging)
            bar_heights = np.zeros(NUM_BARS)
            for i in range(NUM_BARS):
                start = self.bin_indices[i]
                end = self.bin_indices[i + 1]
                if start == end:
                    bar_heights[i] = fft_data[start]
                else:
                    bar_heights[i] = np.mean(fft_data[start:end])

            # 1. Frequenzkompensation anwenden (Höhen boosten, Bass dämpfen)
            bar_heights *= self.eq_weights

            # --- NEUE DYNAMIK-SKALIERUNG ---
            # 2. Exponentielle Skalierung statt starrer Log-Kompression
            
            # Grund-Empfindlichkeit (regelt, wie stark das Eingangssignal generell ist)
            SENSITIVITY = 0.003
            bar_heights = bar_heights * SENSITIVITY
            
            # Exponentielles Skalieren (z.B. hoch 1.3). 
            # Das ist der Zaubertrick gegen die "einheitliche Linie"!
            # Es sorgt dafür, dass Spitzen (laute Beats) nach oben ausbrechen, 
            # während leise Töne verhältnismäßig flach bleiben.
            bar_heights = np.power(bar_heights, 1.3) * 0.001

            # Noise-Gate: Schneidet leises Grundrauschen ab. 
            # Zwingt die Balken bei Stille oder sehr leisen Tönen auf echte 0.
            NOISE_GATE = 4.0
            bar_heights = np.maximum(0, bar_heights - NOISE_GATE)

            # Begrenzung nach oben, damit nichts aus dem Plot (0-120) herausfliegt
            bar_heights = np.clip(bar_heights, 0, 120)

            # --- NEUE GLÄTTUNG ---
            # 3. Asymmetrische Glättung für den "0 auf 100"-Effekt
            ATTACK = 0.5  # Sehr hoch (0.85 - 0.95) -> extrem reaktionsschnell bei Beats ("0 auf 100")
            DECAY = 0.2   # Niedrig (0.1 - 0.2) -> angenehmes, nicht zu hektisches Zurückfallen

            smoothed = np.where(
                bar_heights > self.prev_heights,
                self.prev_heights + (bar_heights - self.prev_heights) * ATTACK,
                self.prev_heights + (bar_heights - self.prev_heights) * DECAY
            )
            
            self.prev_heights = smoothed
            self.bars.setOpts(height=smoothed)

        except Exception as e:
            print(f"Fehler im Update Loop: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    vis = AudioVisualizer()
    vis.show()
    sys.exit(app.exec())