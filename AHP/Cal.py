import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CompleteAHPCalculator:
    def __init__(self, master):
        self.master = master
        master.title("Complete AHP Calculator")
        master.geometry("800x600")

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=1, fill="both")

        self.setup_params_tab()
        self.setup_matrix_tab()
        self.setup_results_tab()

        self.scale_values = [
            "9", "8", "7", "6", "5", "4", "3", "2", "1",
            "1/2", "1/3", "1/4", "1/5", "1/6", "1/7", "1/8", "1/9"
        ]

    def setup_params_tab(self):
        params_frame = ttk.Frame(self.notebook)
        self.notebook.add(params_frame, text="Parameters")

        ttk.Label(params_frame, text="Enter parameters (comma-separated):").pack(pady=5)
        self.params_entry = ttk.Entry(params_frame, width=70)
        self.params_entry.pack(pady=5)
        ttk.Button(params_frame, text="Set Parameters", command=self.set_params).pack(pady=5)

        ttk.Button(params_frame, text="Import from CSV", command=self.import_csv).pack(pady=5)
        ttk.Button(params_frame, text="Export to CSV", command=self.export_csv).pack(pady=5)

    def setup_matrix_tab(self):
        self.matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_frame, text="Comparison Matrix")

        self.add_matrix_example()

    def setup_results_tab(self):
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")

        self.result_text = tk.Text(self.results_frame, height=10, width=70)
        self.result_text.pack(pady=10)

        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.get_tk_widget().pack()

    def add_matrix_example(self):
        example_text = """
填写示例:
1. 从下拉菜单中选择值，范围从 1/9 到 9
2. 含义:
   1: 同等重要
   3: 稍微重要
   5: 明显重要
   7: 强烈重要
   9: 极其重要
   2,4,6,8: 中间值
3. 示例: 如果A比B重要程度是5，则在A行B列选择5，B行A列会自动设置为1/5
4. 对角线元素默认为1，无需填写
5. 只需填写上三角矩阵，下三角矩阵会自动计算
        """
        example_label = ttk.Label(self.matrix_frame, text=example_text, justify=tk.LEFT)
        example_label.pack(pady=10, padx=10, anchor="w")

    def set_params(self):
        params = [p.strip() for p in self.params_entry.get().split(',')]
        if len(params) < 2:
            messagebox.showerror("Error", "Please enter at least two parameters")
            return

        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        self.add_matrix_example()

        canvas = tk.Canvas(self.matrix_frame)
        scrollbar = ttk.Scrollbar(self.matrix_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.comboboxes = []
        for i, param1 in enumerate(params):
            row = []
            for j, param2 in enumerate(params):
                if i == j:
                    label = ttk.Label(scrollable_frame, text="1")
                    label.grid(row=i + 1, column=j + 1)
                elif i < j:
                    combobox = ttk.Combobox(scrollable_frame, values=self.scale_values, width=5)
                    combobox.grid(row=i + 1, column=j + 1)
                    combobox.bind("<<ComboboxSelected>>", lambda e, i=i, j=j: self.update_inverse(i, j))
                    row.append(combobox)
                else:
                    label = ttk.Label(scrollable_frame, text="-")
                    label.grid(row=i + 1, column=j + 1)
            self.comboboxes.append(row)

        for i, param in enumerate(params):
            ttk.Label(scrollable_frame, text=param).grid(row=0, column=i + 1)
            ttk.Label(scrollable_frame, text=param).grid(row=i + 1, column=0)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Button(self.matrix_frame, text="Calculate", command=self.calculate).pack(pady=10)

    def update_inverse(self, i, j):
        value = self.comboboxes[i][j - i - 1].get()
        if value in self.scale_values:
            inverse_index = self.scale_values.index(value)
            inverse_value = self.scale_values[-(inverse_index + 1)]
            if j - i - 1 < len(self.comboboxes[j]) and i < len(self.comboboxes):
                self.comboboxes[j][i].set(inverse_value)
        self.check_consistency()

    def get_ri(self, n):
        ri_table = {
            1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45,
            10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
        }

        if n in ri_table:
            return ri_table[n]
        else:
            return 1.98 * (n - 2) / n

    def check_consistency(self):
        try:
            matrix = self.get_matrix()
            n = len(matrix)
            eigenvalues = np.linalg.eigvals(matrix)
            lambda_max = max(eigenvalues.real)
            ci = (lambda_max - n) / (n - 1)
            ri = self.get_ri(n)
            cr = ci / ri if ri != 0 else 0

            if cr < 0.1:
                self.master.configure(background='green')
            else:
                self.master.configure(background='red')
        except:
            self.master.configure(background='yellow')

    def get_matrix(self):
        n = len(self.comboboxes) + 1
        matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    value = self.comboboxes[i][j - i - 1].get()
                    if '/' in value:
                        num, denom = value.split('/')
                        value = float(num) / float(denom)
                    else:
                        value = float(value)
                    matrix[i, j] = value
                    matrix[j, i] = 1 / value
                except:
                    pass
        return matrix

    def calculate(self):
        try:
            matrix = self.get_matrix()
            n = len(matrix)

            # Calculate the principal eigenvector
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            max_index = np.argmax(eigenvalues.real)
            principal_eigenvector = eigenvectors[:, max_index].real
            weights = principal_eigenvector / np.sum(principal_eigenvector)

            # Calculate Consistency Ratio
            lambda_max = eigenvalues[max_index].real
            ci = (lambda_max - n) / (n - 1)
            ri = self.get_ri(n)
            cr = ci / ri if ri != 0 else 0

            result = "Weights:\n"
            params = [p.strip() for p in self.params_entry.get().split(',')]
            for param, weight in zip(params, weights):
                result += f"{param}: {weight:.4f}\n"
            result += f"\nCI: {ci:.4f}\n"
            result += f"RI: {ri:.4f}\n"
            result += f"CR: {cr:.4f}\n"
            result += "Consistency: " + ("Acceptable" if cr < 0.1 else "Not acceptable")

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)

            self.plot_results(params, weights)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_results(self, params, weights):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.bar(params, weights)
        ax.set_ylabel('Weight')
        ax.set_title('AHP Results')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        self.fig.tight_layout()
        self.canvas.draw()

    def import_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path, index_col=0)
                params = df.index.tolist()
                self.params_entry.delete(0, tk.END)
                self.params_entry.insert(0, ", ".join(params))
                self.set_params()
                for i in range(len(params)):
                    for j in range(i + 1, len(params)):
                        value = df.iloc[i, j]
                        if value != 1:
                            if value < 1:
                                value = f"1/{int(1/value)}"
                            else:
                                value = str(int(value))
                            self.comboboxes[i][j - i - 1].set(value)
                self.check_consistency()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import CSV: {str(e)}")

    def export_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                matrix = self.get_matrix()
                params = [p.strip() for p in self.params_entry.get().split(',')]
                df = pd.DataFrame(matrix, index=params, columns=params)
                df.to_csv(file_path)
                messagebox.showinfo("Success", "Data exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV: {str(e)}")


def main():
    root = tk.Tk()
    app = CompleteAHPCalculator(root)
    root.mainloop()


if __name__ == "__main__":
    main()