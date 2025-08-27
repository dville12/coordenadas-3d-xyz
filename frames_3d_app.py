# -*- coding: utf-8 -*-
"""
Aplicación 3D para graficar sistemas de coordenadas encadenados:
W -> S0 -> S1 -> S2

- Campo 1: S0 en W
- Campo 2: S1 en S0
- Campo 3: S2 en S1

Rotaciones en orden Euler XYZ: R = Rx * Ry * Rz (grados).
Longitud y color de ejes configurables por sistema (W, S0, S1, S2).
Autor: Daniel + M365 Copilot
"""

import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QPushButton, QFrame, QMessageBox, QColorDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ---------------------- Utilidades de álgebra ----------------------

def deg2rad(d):
    return np.deg2rad(d)


def rot_xyz(rx_deg, ry_deg, rz_deg):
    """
    Construye R = Rx(rx) * Ry(ry) * Rz(rz)
    Ángulos en grados.
    """
    x = deg2rad(rx_deg)
    y = deg2rad(ry_deg)
    z = deg2rad(rz_deg)

    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    Rx = np.array([[1,  0,   0],
                   [0, cx, -sx],
                   [0, sx,  cx]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])

    return Rx @ Ry @ Rz


def T_from_txyzrxyz(tx, ty, tz, rx, ry, rz):
    """Matriz homogénea 4x4 desde traslación (tx,ty,tz) y Euler XYZ (rx,ry,rz)."""
    R = rot_xyz(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = np.array([tx, ty, tz])
    return T


def format_matrix(T, name):
    """Devuelve texto formateado de la matriz homogénea."""
    A = np.array(T)
    with np.printoptions(precision=5, suppress=True):
        return f"{name} =\n{A}\n"


def qcolor_to_rgb(qc: QColor):
    """Convierte QColor a tupla RGB normalizada (0..1)."""
    return (qc.redF(), qc.greenF(), qc.blueF())


# ---------------------- Lienzo Matplotlib 3D ----------------------

class MplCanvas3D(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.view_init(elev=25, azim=45)
        self.fig.tight_layout()

    def set_equal_limits_auto(self, points, Ls):
        """
        Ajusta límites iguales en 3D a partir de lista de puntos (Nx3)
        y longitudes de ejes (lista Ls) para margen.
        """
        P = np.array(points) if len(points) else np.zeros((1, 3))
        mins = P.min(axis=0)
        maxs = P.max(axis=0)
        center = (mins + maxs) / 2.0
        base = max((maxs - mins).max(), max(Ls) if Ls else 1.0)
        r = base * 1.2
        xlim = (center[0] - r, center[0] + r)
        ylim = (center[1] - r, center[1] + r)
        zlim = (center[2] - r, center[2] + r)
        self.ax.set_xlim(xlim); self.ax.set_ylim(ylim); self.ax.set_zlim(zlim)
        self.ax.set_box_aspect([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])

    def draw_frame(self, T, name="S", L=1.0, color=(0.2, 0.2, 0.2), alpha=1.0, lw=2.2):
        """
        Dibuja los ejes del marco T usando un único color por sistema (según lo solicitado).
        """
        o = T[:3, 3]
        R = T[:3, :3]
        x = o + R[:, 0] * L
        y = o + R[:, 1] * L
        z = o + R[:, 2] * L

        # ejes (mismo color para las 3 direcciones)
        self.ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], color=color, lw=lw, alpha=alpha)
        self.ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], color=color, lw=lw, alpha=alpha)
        self.ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], color=color, lw=lw, alpha=alpha)
        # origen
        self.ax.scatter([o[0]], [o[1]], [o[2]], color=color, s=24, alpha=alpha)
        # etiqueta
        self.ax.text(o[0], o[1], o[2], f" {name}", color=color, fontsize=10, alpha=alpha, weight='bold')


# ---------------------- UI principal ----------------------

class SpinTriple(QWidget):
    """Conjunto de 3 QDoubleSpinBox etiquetados (x, y, z) o (rotX, rotY, rotZ)."""
    def __init__(self, labels=('x', 'y', 'z'), decimals=3, step=0.1, range_min=-1e6, range_max=1e6):
        super().__init__()
        grid = QGridLayout(self)
        self.spins = []
        for i, lab in enumerate(labels):
            grid.addWidget(QLabel(lab), 0, i)
            sb = QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setRange(range_min, range_max)
            sb.setSingleStep(step)
            sb.setValue(0.0)
            grid.addWidget(sb, 1, i)
            self.spins.append(sb)

    def values(self):
        return [sb.value() for sb in self.spins]

    def set_values(self, vals):
        for sb, v in zip(self.spins, vals):
            sb.setValue(v)


class FrameInput(QGroupBox):
    """Grupo de entrada para un marco: traslación y Euler XYZ."""
    def __init__(self, title):
        super().__init__(title)
        lay = QVBoxLayout(self)
        self.tr = SpinTriple(('tx', 'ty', 'tz'), decimals=3, step=0.1, range_min=-1e6, range_max=1e6)
        self.rxyz = SpinTriple(('rotX°', 'rotY°', 'rotZ°'), decimals=2, step=1.0, range_min=-360, range_max=360)
        lay.addWidget(QLabel("Traslación"))
        lay.addWidget(self.tr)
        lay.addWidget(QLabel("Rotación Euler (XYZ, grados)"))
        lay.addWidget(self.rxyz)

    def get_T(self):
        tx, ty, tz = self.tr.values()
        rx, ry, rz = self.rxyz.values()
        return T_from_txyzrxyz(tx, ty, tz, rx, ry, rz)

    def reset(self):
        self.tr.set_values((0.0, 0.0, 0.0))
        self.rxyz.set_values((0.0, 0.0, 0.0))


class AppearanceRow(QWidget):
    """
    Fila para configurar longitud y color de un sistema.
    """
    def __init__(self, name, default_L=1.0, default_color="#666666"):
        super().__init__()
        lay = QHBoxLayout(self)
        self.name = name
        lay.addWidget(QLabel(name))
        lay.addWidget(QLabel("L:"))
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setDecimals(2)
        self.spin_L.setRange(0.01, 1e6)
        self.spin_L.setSingleStep(0.1)
        self.spin_L.setValue(default_L)
        lay.addWidget(self.spin_L)

        self.btn_color = QPushButton("Color")
        self.color = QColor(default_color)
        self._apply_btn_style()
        self.btn_color.clicked.connect(self.pick_color)
        lay.addWidget(self.btn_color)
        lay.addStretch(1)

    def pick_color(self):
        c = QColorDialog.getColor(self.color, self, f"Color {self.name}")
        if c.isValid():
            self.color = c
            self._apply_btn_style()

    def _apply_btn_style(self):
        self.btn_color.setStyleSheet(f"background-color: {self.color.name()}; color: white;")

    def get_L_color(self):
        return self.spin_L.value(), qcolor_to_rgb(self.color)

    def set_L(self, value):
        self.spin_L.setValue(value)

    def set_color_hex(self, hex_color):
        self.color = QColor(hex_color)
        self._apply_btn_style()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coordenadas 3D (XYZ): W → S0 → S1 → S2")

        # --- panel izquierdo (controles) ---
        left = QWidget()
        left_lay = QVBoxLayout(left)

        self.g0 = FrameInput("Campo 1: Sistema 0 (S0) en W")
        self.g1 = FrameInput("Campo 2: Sistema 1 (S1) en S0")
        self.g2 = FrameInput("Campo 3: Sistema 2 (S2) en S1")

        # Apariencia por sistema
        app_box = QGroupBox("Apariencia (longitud y color por sistema)")
        app_lay = QVBoxLayout(app_box)
        # valores por defecto
        self.ap_W  = AppearanceRow("W",  default_L=1.10, default_color="#666666")
        self.ap_S0 = AppearanceRow("S0", default_L=1.00, default_color="#E53935")  # rojo
        self.ap_S1 = AppearanceRow("S1", default_L=1.00, default_color="#43A047")  # verde
        self.ap_S2 = AppearanceRow("S2", default_L=1.00, default_color="#1E88E5")  # azul
        for row in (self.ap_W, self.ap_S0, self.ap_S1, self.ap_S2):
            app_lay.addWidget(row)

        # Botones
        btns = QHBoxLayout()
        self.btn_update = QPushButton("Actualizar")
        self.btn_reset = QPushButton("Reset")
        self.btn_copy = QPushButton("Copiar matrices")
        btns.addWidget(self.btn_update)
        btns.addWidget(self.btn_reset)
        btns.addWidget(self.btn_copy)

        left_lay.addWidget(self.g0)
        left_lay.addWidget(self.g1)
        left_lay.addWidget(self.g2)
        left_lay.addWidget(app_box)
        left_lay.addLayout(btns)
        left_lay.addStretch(1)

        # --- panel derecho (gráfico) ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        self.canvas = MplCanvas3D()
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_lay.addWidget(self.toolbar)
        right_lay.addWidget(self.canvas)

        # --- splitter horizontal ---
        central = QWidget()
        main_lay = QHBoxLayout(central)
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)
        main_lay.addWidget(left, 0)
        main_lay.addWidget(vline)
        main_lay.addWidget(right, 1)
        self.setCentralWidget(central)

        # conexiones
        self.btn_update.clicked.connect(self.update_plot)
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_copy.clicked.connect(self.copy_matrices)

        # valores de ejemplo
        self.g0.tr.set_values((0.0, 0.0, 0.0))
        self.g0.rxyz.set_values((0.0, 0.0, 0.0))
        self.g1.tr.set_values((1.0, 0.0, 0.0))
        self.g1.rxyz.set_values((45.0, 0.0, 0.0))  # rotX
        self.g2.tr.set_values((0.0, 1.0, 0.0))
        self.g2.rxyz.set_values((0.0, 0.0, 30.0))  # rotZ

        self.update_plot()

    def compute_transforms(self):
        """Devuelve T_W_S0, T_S0_S1, T_S1_S2, T_W_S1, T_W_S2."""
        T_W_S0 = self.g0.get_T()
        T_S0_S1 = self.g1.get_T()
        T_S1_S2 = self.g2.get_T()
        T_W_S1 = T_W_S0 @ T_S0_S1
        T_W_S2 = T_W_S0 @ T_S0_S1 @ T_S1_S2
        return T_W_S0, T_S0_S1, T_S1_S2, T_W_S1, T_W_S2

    def update_plot(self):
        ax = self.canvas.ax
        # conservar vista actual
        elev, azim = ax.elev, ax.azim
        ax.cla()
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)

        # Transformaciones
        T_W_S0, T_S0_S1, T_S1_S2, T_W_S1, T_W_S2 = self.compute_transforms()

        # Apariencia
        L_W,  C_W  = self.ap_W.get_L_color()
        L_0,  C_0  = self.ap_S0.get_L_color()
        L_1,  C_1  = self.ap_S1.get_L_color()
        L_2,  C_2  = self.ap_S2.get_L_color()

        # Dibujar
        I = np.eye(4)
        self.canvas.draw_frame(I,        "W",  L=L_W, color=C_W, alpha=0.8, lw=1.8)
        self.canvas.draw_frame(T_W_S0,   "S0", L=L_0, color=C_0, alpha=1.0)
        self.canvas.draw_frame(T_W_S1,   "S1", L=L_1, color=C_1, alpha=0.95)
        self.canvas.draw_frame(T_W_S2,   "S2", L=L_2, color=C_2, alpha=0.95)

        # Auto límites usando orígenes y longitudes
        points = [np.zeros(3), T_W_S0[:3, 3], T_W_S1[:3, 3], T_W_S2[:3, 3]]
        self.canvas.set_equal_limits_auto(points, [L_W, L_0, L_1, L_2])
        self.canvas.draw()

    def reset_all(self):
        self.g0.reset(); self.g1.reset(); self.g2.reset()
        self.ap_W.set_L(1.10);  self.ap_W.set_color_hex("#666666")
        self.ap_S0.set_L(1.00); self.ap_S0.set_color_hex("#E53935")
        self.ap_S1.set_L(1.00); self.ap_S1.set_color_hex("#43A047")
        self.ap_S2.set_L(1.00); self.ap_S2.set_color_hex("#1E88E5")
        self.update_plot()

    def copy_matrices(self):
        T_W_S0, T_S0_S1, T_S1_S2, T_W_S1, T_W_S2 = self.compute_transforms()
        text = "Convención de rotación: Euler XYZ (R = Rx * Ry * Rz, grados)\n\n"
        text += format_matrix(T_W_S0, "T_W_S0")
        text += format_matrix(T_S0_S1, "T_S0_S1")
        text += format_matrix(T_S1_S2, "T_S1_S2")
        text += format_matrix(T_W_S1, "T_W_S1 = T_W_S0 @ T_S0_S1")
        text += format_matrix(T_W_S2, "T_W_S2 = T_W_S0 @ T_S0_S1 @ T_S1_S2")
        QGuiApplication.clipboard().setText(text)
        QMessageBox.information(self, "Matrices copiadas", "Se copiaron las matrices al portapapeles.")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1180, 680)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
