import tkinter as tk
import customtkinter as ctk
import pandas as pd
import numpy as np
from itertools import tee, islice, chain
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class App(ctk.CTk):
  def __init__(self):
    super().__init__()
    self.title("Markov Chain")
    self.steady_state = None
    self.window_width = 800
    self.window_height = 600
    self.data = None
    self.has_steady_state = lambda: self.steady_state is not None
    self.grid_rowconfigure(0, weight=1)
    self.grid_columnconfigure(0, weight=1)
    self._init_component()
    self._center_window()

  def _init_component(self):
    self.menu_frame = ctk.CTkFrame(self, corner_radius=0)
    self.menu_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    self.open_button = ctk.CTkButton(
      self.menu_frame, text="Open", command=self._open_file, corner_radius=0)
    self.open_button.pack(anchor="center", side=tk.LEFT,
                          expand=True, padx=10, pady=10)

    self.credit_button = ctk.CTkButton(
      self.menu_frame, text="Credits", command=self._show_credits, corner_radius=0)
    self.credit_button.pack(anchor="center", side=tk.LEFT,
                            expand=True, padx=10, pady=10)

    self.tab_view = ctk.CTkTabview(self, corner_radius=0)
    self.tab_view.pack(side=tk.TOP, fill=tk.BOTH,
                       expand=True, padx=10, pady=(0, 10))
    self.tab_view.add("Data")
    self.tab_view.add("Visualisasi")
    self.tab_view.add("Prediksi")
    self._init_prediksi()

  def _center_window(self, frame=None):
    if frame is not None:
      screen_width = frame.winfo_screenwidth()
      screen_height = frame.winfo_screenheight()
      x = (screen_width - frame.window_width) // 2
      y = (screen_height - frame.window_height) // 2

      frame.geometry(f"{frame.window_width}x{frame.window_height}+{x}+{y}")
    else:
      screen_width = self.winfo_screenwidth()
      screen_height = self.winfo_screenheight()
      x = (screen_width - self.window_width) // 2
      y = (screen_height - self.window_height) // 2

      self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

  def _clear_data(self):
    for widget in self.tab_view.tab("Data").winfo_children():
      widget.destroy()
    self.frame_table.pack_forget()
    self.frame_markov.pack_forget()

  def _init_prediksi(self):
    self.frame_prediksi = ctk.CTkFrame(
      self.tab_view.tab("Prediksi"), corner_radius=0)
    self.frame_prediksi.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)
    self.input_prediksi = ctk.CTkEntry(
      self.frame_prediksi, placeholder_text=f"masukan tahun", corner_radius=0)
    self.input_prediksi.pack(side=tk.LEFT, fill=tk.BOTH,
                             expand=True, padx=10, pady=5)
    self.button_prediksi = ctk.CTkButton(
      self.frame_prediksi, text="predict", command=self._prediksi, corner_radius=0)
    self.button_prediksi.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=5)
    pass

  def _init_table(self):
    if self.data is not None:
      # Data
      self.frame_table = ctk.CTkScrollableFrame(
        self.tab_view.tab("Data"), corner_radius=0)
      self.frame_table.pack(side=tk.LEFT, anchor=tk.W,
                            fill=tk.BOTH, padx=10, pady=10)

      for i, header in enumerate(self.data.columns):
        label = ctk.CTkLabel(self.frame_table, text=header,
                             fg_color="transparent")
        label.grid(row=0, column=i, sticky=tk.NSEW, padx=8, pady=2)

      for row, row_data in enumerate(self.data.values, start=1):
        for column, cell_data in enumerate(row_data):
          label = ctk.CTkLabel(self.frame_table, text=int(
            cell_data) if column == 0 else cell_data)
          label.grid(row=row, column=column, sticky=tk.NSEW)

      # Prob A & Matrik Transisi
      self.frame_markov = ctk.CTkFrame(
        self.tab_view.tab("Data"), corner_radius=0)
      self.frame_markov.pack(side=tk.TOP, anchor=tk.CENTER,
                             fill=tk.BOTH, padx=10, pady=10)

      self.ctklabel_proba = ctk.CTkLabel(
        self.frame_markov, text="Probabilitas Awal")
      self.ctklabel_proba.pack(
        side=tk.TOP, anchor=tk.CENTER, fill=tk.BOTH, padx=10, pady=10)
      self.frame_proba = ctk.CTkFrame(self.frame_markov, corner_radius=0)
      self.frame_proba.pack(side=tk.TOP, anchor=tk.CENTER,
                            fill=tk.BOTH, padx=10, pady=10)
      for row, row_data in enumerate([self.label_proba, self.proba]):
        for column, cell_data in enumerate(row_data):
          label = ctk.CTkLabel(self.frame_proba, text=cell_data)
          label.grid(row=row, column=column, sticky="ew", padx=20)

      self.ctklabel_matrixtrans = ctk.CTkLabel(
        self.frame_markov, text="Matrix Transisi")
      self.ctklabel_matrixtrans.pack(
        side=tk.TOP, anchor=tk.CENTER, fill=tk.BOTH, padx=10, pady=10)
      self.frame_matrixtrans = ctk.CTkFrame(self.frame_markov, corner_radius=0)
      self.frame_matrixtrans.pack(
        side=tk.TOP, anchor=tk.CENTER, fill=tk.BOTH, padx=10, pady=10)
      for i, header in enumerate([" ", *self.label_proba]):
        label = ctk.CTkLabel(self.frame_matrixtrans,
                             text=header, fg_color="transparent")
        label.grid(row=0, column=i, sticky=tk.NSEW, padx=8, pady=2)
      for row, row_data in enumerate(np.c_[self.label_proba, self.m_transisi], start=1):
        for column, cell_data in enumerate(row_data):
          label = ctk.CTkLabel(self.frame_matrixtrans, text=cell_data)
          label.grid(row=row, column=column, sticky="ew", padx=20)

      # Steady State
      self.ctklabel_steadystate = ctk.CTkLabel(
        self.frame_markov, text="Steady State")
      self.ctklabel_steadystate.pack(
        side=tk.TOP, anchor=tk.CENTER, fill=tk.BOTH, padx=10, pady=10)
      self.frame_steadystate = ctk.CTkFrame(self.frame_markov, corner_radius=0)
      self.frame_steadystate.pack(
        side=tk.TOP, anchor=tk.CENTER, fill=tk.BOTH, padx=10, pady=10)
      if not self.has_steady_state():
        ss_404 = ctk.CTkLabel(self.frame_steadystate,
                              text="Steady State Not Found")
        ss_404.grid(row=0, column=0, sticky="ew", padx=20)
      else:
        for column, cell_data in enumerate(self.steady_state):
          label = ctk.CTkLabel(self.frame_steadystate, text=str(cell_data))
          label.grid(row=0, column=column, sticky="ew", padx=20)

  def _prediksi(self):
    input = self.input_prediksi.get()
    if int(input) > self.tahun[-1]:
      max_state = abs(self.tahun[-1] - int(input))
      _, all_prob = self._find_steady_state(
        max_iteration=max_state, get_all=True)
      if hasattr(self, 'all_prob_frame'):
        for widget in self.all_prob_frame.winfo_children():
          widget.destroy()
        self.all_prob_frame.pack_forget()
      self.all_prob_frame = ctk.CTkFrame(
        self.tab_view.tab("Prediksi"), corner_radius=0)
      self.all_prob_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=5)

      for i, header in enumerate(["Tahun\\Predikat", *self.label_proba]):
        label = ctk.CTkLabel(self.all_prob_frame,
                             text=header, fg_color="transparent")
        label.grid(row=0, column=i, sticky=tk.NSEW, padx=8, pady=2)

      tahun = self.tahun[-1]

      for row, row_data in enumerate(all_prob[:int(max_state)], start=1):
        label = ctk.CTkLabel(self.all_prob_frame, text=int(tahun + row))
        label.grid(row=row, column=0, sticky="ew", padx=20)
        for column, cell_data in enumerate(row_data, start=1):
          label = ctk.CTkLabel(self.all_prob_frame,
                               text=f"{round(cell_data*100,2)}%")
          label.grid(row=row, column=column, sticky="ew", padx=20)
    else:
      idx = np.where(self.tahun == int(input))[0]
      tk.messagebox.showinfo(
        f"Tahun {input}", f"Produksi : {self.produksi[idx]}")

  def _open_file(self):
    filetypes = [("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")]
    filepath = tk.filedialog.askopenfilename(
      title="Open File", filetypes=filetypes)
    if filepath:
      if filepath.endswith(".xlsx"):
        self.data = pd.read_excel(filepath)
      elif filepath.endswith(".csv"):
        self.data = pd.read_csv(filepath)
      self.tahun = self.data.values[:, 0]
      self.produksi = self.data.values[:, 1]
      self._show_data()
      self._show_plot()

  def _show_data(self):
    self._labeling()
    print(self.label)
    self._probabilitas_awal()
    print(self.proba)
    self._matrix_transisi()
    print(self.m_transisi)
    self._find_steady_state()
    if len(self.tab_view.tab("Data").winfo_children()) > 0:
      self._clear_data()
    self._init_table()

  def _show_plot(self):
    fig = Figure(figsize=(5, 5), dpi=100)
    Y = self.produksi
    X = self.tahun
    plotting = fig.add_subplot(111)
    plotting.plot(X, Y)
    plotting.set_ylabel("Produksi")
    plotting.set_xlabel("Tahun")
    canvas = FigureCanvasTkAgg(fig, master=self.tab_view.tab("Visualisasi"))
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)
    pass

  def _show_credits(self):
    tk.messagebox.showinfo("Credits", "Kurniawan // Kristiawan // Vina")

  def __previous_and_next(self, some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

  def _labeling(self):
    produksi = self.produksi
    min, max = produksi.min(), produksi.max()
    range_data = (max - min) / 3
    label = []

    for x in produksi:
      if x < min + range_data:
        label.append('S')
      elif min + range_data <= x < min + range_data * 2:
        label.append('C')
      elif x > min + range_data * 2:
        label.append('B')

    self.label = np.array(label)
    self.data['label'] = self.label
    return self.label

  def _probabilitas_awal(self):
    unique_val, count = np.unique(self.label, return_counts=True)
    count = [round(x / count.sum(), 4) for x in count]
    self.label_proba = np.flip(unique_val)
    self.proba = np.flip(count)
    return self.label_proba, self.proba

  def _matrix_transisi(self):
    transisi = np.zeros((3, 3))

    for _, item, nxt in self.__previous_and_next(self.label):
      if nxt is None:
        continue
      ps = ['S', 'C', 'B'].index(item)
      pd = ['S', 'C', 'B'].index(nxt)
      transisi[ps][pd] += 1

    for i, y in enumerate(transisi):
      row_total = y.sum()
      for j, x in enumerate(y):
        transisi[i][j] = round(x / row_total, 4)

    self.m_transisi = transisi
    return self.m_transisi

  def _find_steady_state(self, max_iteration: int = 100, get_all: bool = False):
    new_prob = [round(x, 4) for x in np.matmul(self.proba, self.m_transisi)]
    prev_val, streak, i = new_prob, 0, 0
    all_prob = [new_prob]
    while i < max_iteration:
      i += 1
      new_prob = [round(x, 4) for x in np.matmul(new_prob, self.m_transisi)]
      all_prob = [*all_prob, new_prob]
      if np.all(np.equal(prev_val, new_prob)):
        streak += 1
      else:
        streak = 0

      if streak > 3:
        if get_all:
          return True, all_prob
        self.steady_state = new_prob
        return True, new_prob

      prev_val = new_prob

    if get_all:
      return False, all_prob
    self.steady_state = None
    return False, None


if __name__ == "__main__":
  app = App()
  app.mainloop()
  # ToplevelWindow().mainloop()
