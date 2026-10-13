import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import OrderedDict

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

class ColorPickerDialog:
    """科研配色选择器对话框"""
    def __init__(self, parent, current_color, color_schemes):
        # 中文配色方案名称映射
        self.scheme_names_zh = {
            'ColorBrewer Set1': 'ColorBrewer 集合1',
            'ColorBrewer Set2': 'ColorBrewer 集合2',
            'ColorBrewer Set3': 'ColorBrewer 集合3',
            'ColorBrewer Dark2': 'ColorBrewer 深色2',
            'ColorBrewer Paired': 'ColorBrewer 配对',
            'Tableau 10': 'Tableau 10色',
            'Tableau 20': 'Tableau 20色',
            'Okabe-Ito (Colorblind)': 'Okabe-Ito (色盲友好)',
            'Wong (Colorblind)': 'Wong (色盲友好)',
            'Tol Bright': 'Tol 明亮',
            'Tol Vibrant': 'Tol 鲜艳',
            'Nature Style': 'Nature 风格',
            'Science Style': 'Science 风格',
            'IEEE Standard': 'IEEE 标准',
            'Matplotlib Default': 'Matplotlib 默认',
            'Seaborn Deep': 'Seaborn 深色',
            'Seaborn Colorblind': 'Seaborn 色盲友好',
            'Viridis': 'Viridis',
            'Plasma': 'Plasma',
            'Inferno': 'Inferno',
            'Magma': 'Magma',
            'Cividis': 'Cividis'
        }
        self.parent = parent
        self.current_color = current_color
        self.color_schemes = color_schemes
        self.selected_color = current_color
        self.selected_scheme = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Select Research Color Scheme")
        self.dialog.geometry("550x500")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 配色方案选择
        ttk.Label(main_frame, text="Color Scheme:").pack(anchor=tk.W, pady=(0,5))

        self.scheme_var = tk.StringVar()
        # 创建中文名称到英文键的反向映射
        self.scheme_names_reverse = {v: k for k, v in self.scheme_names_zh.items()}

        # 默认选中第一个配色方案
        first_scheme_zh = list(self.scheme_names_zh.values())[0]
        self.scheme_var.set(first_scheme_zh)

        self.scheme_combo = ttk.Combobox(
            main_frame,
            textvariable=self.scheme_var,
            values=list(self.scheme_names_zh.values()),
            state='readonly'
        )
        self.scheme_combo.pack(fill=tk.X, pady=(0,10))
        self.scheme_combo.bind('<<ComboboxSelected>>', self.on_scheme_selected)

        # 颜色预览区域
        ttk.Label(main_frame, text="Color Preview:").pack(anchor=tk.W, pady=(0,5))

        # 创建带滚动条的框架
        self.color_canvas_frame = ttk.Frame(main_frame)
        self.color_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))

        # 创建画布和滚动条
        self.color_canvas = tk.Canvas(self.color_canvas_frame, height=200, width=280)
        self.color_scrollbar = ttk.Scrollbar(self.color_canvas_frame, orient=tk.VERTICAL, command=self.color_canvas.yview)
        self.color_canvas.configure(yscrollcommand=self.color_scrollbar.set)

        # 放置画布和滚动条
        self.color_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.color_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建颜色按钮的容器框架
        self.color_frame = ttk.Frame(self.color_canvas)
        self.color_canvas.create_window((0, 0), window=self.color_frame, anchor=tk.NW)

        # 绑定鼠标滚轮事件
        self.color_canvas.bind('<Configure>', self.on_canvas_configure)
        self.color_canvas.bind_all('<MouseWheel>', self.on_mousewheel)

        self.color_buttons = []
        self.create_color_buttons()

        # 当前选择显示
        current_frame = ttk.Frame(main_frame)
        current_frame.pack(fill=tk.X, pady=(10,0))

        ttk.Label(current_frame, text="Current Selection:").pack(side=tk.LEFT)
        self.current_label = ttk.Label(current_frame, text="", foreground=self.current_color)
        self.current_label.pack(side=tk.LEFT, padx=(10,0))

        # 自定义颜色输入
        custom_frame = ttk.Frame(main_frame)
        custom_frame.pack(fill=tk.X, pady=(10,0))

        ttk.Label(custom_frame, text="自定义颜色 (十六进制):").pack(side=tk.LEFT)
        self.custom_color_var = tk.StringVar()
        self.custom_color_entry = ttk.Entry(custom_frame, textvariable=self.custom_color_var, width=10)
        self.custom_color_entry.pack(side=tk.LEFT, padx=(5,0))
        # 设置占位符文本，提示用户输入格式
        self.custom_color_entry.insert(0, "#000000")
        self.custom_color_entry.config(foreground='gray')

        # 绑定事件，当用户开始输入时清除占位符
        def on_entry_focus(event):
            if self.custom_color_entry.get() == "#000000":
                self.custom_color_entry.delete(0, tk.END)
                self.custom_color_entry.config(foreground='black')

        def on_entry_focusout(event):
            if not self.custom_color_entry.get():
                self.custom_color_entry.insert(0, "#000000")
                self.custom_color_entry.config(foreground='gray')

        self.custom_color_entry.bind('<FocusIn>', on_entry_focus)
        self.custom_color_entry.bind('<FocusOut>', on_entry_focusout)

        ttk.Button(custom_frame, text="应用自定义", command=self.apply_custom_color).pack(side=tk.LEFT, padx=(5,0))

        # 按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(20,0))

        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.RIGHT, padx=(5,0))
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT)

    def create_color_buttons(self):
        """创建颜色按钮网格"""
        # 清除之前的按钮和行容器，防止残留的空白行
        for btn in self.color_buttons:
            try:
                btn.destroy()
            except Exception:
                pass
        self.color_buttons.clear()
        for child in self.color_frame.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass

        # 获取当前选中的配色方案的颜色列表
        selected_scheme_zh = self.scheme_var.get()
        if selected_scheme_zh:
            selected_scheme_en = self.scheme_names_reverse.get(selected_scheme_zh, selected_scheme_zh)
            colors = self.color_schemes.get(selected_scheme_en, [])
        else:
            # 如果没有选中方案，默认使用第一个方案
            first_scheme = list(self.color_schemes.keys())[0]
            colors = self.color_schemes.get(first_scheme, [])

        # 根据颜色数量动态调整网格布局
        if len(colors) <= 9:
            # 3x3网格
            rows, cols = 3, 3
        elif len(colors) <= 12:
            # 4x3网格
            rows, cols = 4, 3
        elif len(colors) <= 15:
            # 5x3网格
            rows, cols = 5, 3
        else:
            # 6x3网格，超过15个颜色时使用6x3布局
            rows, cols = 6, 3

        # 创建网格
        for i in range(rows):
            row_frame = ttk.Frame(self.color_frame)
            row_frame.pack(fill=tk.X, pady=1, anchor=tk.W)  # 确保左对齐

            for j in range(cols):
                idx = i * cols + j
                if idx < len(colors):
                    color = colors[idx]
                    btn = tk.Button(
                        row_frame,
                        bg=color,
                        text="",  # 不显示文字，只显示颜色
                        width=8,  # 稍微缩小宽度
                        height=2,
                        relief=tk.RAISED,
                        command=lambda c=color: self.select_color(c)
                    )
                    btn.pack(side=tk.LEFT, padx=1, pady=1)
                    self.color_buttons.append(btn)

        # 更新画布滚动区域：使用容器的请求尺寸（winfo_reqwidth/height）来避免空白
        self.color_frame.update_idletasks()

        try:
            req_w = self.color_frame.winfo_reqwidth()
            req_h = self.color_frame.winfo_reqheight()
            # 设置滚动区域并定位到顶部，添加少量边距
            self.color_canvas.configure(scrollregion=(0, 0, req_w + 10, req_h + 10))

            # 延迟多次将视图移动到顶部，确保不同平台/主题下都生效
            def _reset_view():
                try:
                    self.color_canvas.yview_moveto(0.0)
                    self.color_canvas.xview_moveto(0.0)
                    # 同步更新小部件布局
                    self.color_canvas.update_idletasks()
                except Exception:
                    pass

            # 多阶段复位：短延迟与较大延迟一起使用，增强可靠性
            self.color_canvas.after(10, _reset_view)
            self.color_canvas.after(80, _reset_view)
            self.color_canvas.after(200, _reset_view)
        except Exception:
            # 兜底默认值
            self.color_canvas.configure(scrollregion=(0, 0, 280, 200))
            self.color_canvas.after(50, lambda: (self.color_canvas.yview_moveto(0.0), self.color_canvas.xview_moveto(0.0)))

    def on_scheme_selected(self, event):
        """配色方案选择事件"""
        chinese_name = self.scheme_var.get()
        self.selected_scheme = self.scheme_names_reverse.get(chinese_name, chinese_name)
        self.create_color_buttons()
        # 再次触发复位，防止在某些系统/主题下滚动条位置未复位的问题
        try:
            self.color_canvas.after(10, lambda: (self.color_canvas.yview_moveto(0.0), self.color_canvas.xview_moveto(0.0)))
            self.color_canvas.after(100, lambda: (self.color_canvas.yview_moveto(0.0), self.color_canvas.xview_moveto(0.0)))
        except Exception:
            pass

    def select_color(self, color):
        """选择颜色"""
        self.selected_color = color
        self.current_label.config(text=color, foreground=color)

        # 高亮显示选中的颜色
        for btn in self.color_buttons:
            if btn.cget('bg') == color:
                btn.config(relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)

    def on_canvas_configure(self, event):
        """画布配置事件"""
        self.color_canvas.configure(scrollregion=self.color_canvas.bbox('all'))

    def on_mousewheel(self, event):
        """鼠标滚轮事件"""
        self.color_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')

    def apply_custom_color(self):
        """应用自定义颜色"""
        custom_color = self.custom_color_var.get().strip()

        # 如果输入的是占位符文本，忽略
        if custom_color == "#000000" and self.custom_color_entry.cget('foreground') == 'gray':
            messagebox.showwarning("输入提示", "请输入一个十六进制颜色代码，例如 #FF0000")
            return

        # 移除所有空格和可能的额外字符
        custom_color = custom_color.replace(' ', '').upper()

        # 如果没有#号，添加它
        if not custom_color.startswith('#'):
            custom_color = '#' + custom_color

        # 验证颜色格式
        if len(custom_color) == 7:
            # 检查是否只包含有效的十六进制字符
            hex_part = custom_color[1:]
            valid_chars = '0123456789ABCDEF'

            if all(c in valid_chars for c in hex_part):
                try:
                    # 测试颜色是否有效
                    self.color_canvas.create_rectangle(0, 0, 1, 1, fill=custom_color)
                    self.select_color(custom_color)
                    return  # 成功应用颜色
                except tk.TclError:
                    pass  # 颜色无效，继续到错误提示

        # 如果到达这里，说明输入无效
        messagebox.showerror("无效颜色", "请输入有效的十六进制颜色代码。\n\n格式：# + 6位十六进制数字\n示例：#FF0000 (红色)、#00FF00 (绿色)、#0000FF (蓝色)")

    def on_ok(self):
        """确定按钮"""
        self.dialog.destroy()

    def on_cancel(self):
        """取消按钮"""
        self.selected_color = self.current_color
        self.dialog.destroy()

    def show(self):
        """显示对话框并返回选择结果"""
        self.dialog.wait_window()
        return self.selected_color, self.selected_scheme

class MultiCurveChartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Curve Data Visualization Tool")
        
        # 存储多个文件的数据
        self.file_data = OrderedDict()  # 文件名 -> 数据字典
        self.file_labels = {}  # 文件名 -> 自定义标签                                                
        self.file_colors = {}  # 文件名 -> 颜色
        
        # 科研标准配色方案（扩展版）
        self.color_schemes = {
            'ColorBrewer Set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb'],
            'ColorBrewer Set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072'],
            'ColorBrewer Set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'],
            'ColorBrewer Dark2': ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666', '#1f78b4', '#33a02c', '#fb9a99', '#fdbf6f'],
            'ColorBrewer Paired': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'],
            'Tableau 10': ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'],
            'Tableau 20': ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'Okabe-Ito (Colorblind)': ['#000000', '#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7', '#999999', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
            'Wong (Colorblind)': ['#000000', '#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7', '#cccccc', '#882255', '#332288', '#117733', '#44aa99', '#999933', '#aa4499'],
            'Tol Bright': ['#4477aa', '#ee6677', '#228833', '#ccbb44', '#66ccee', '#aa3377', '#bbbbbb', '#ee7733', '#009988', '#eecc66', '#994455', '#997700', '#999999'],
            'Tol Vibrant': ['#0077bb', '#33bbee', '#009988', '#ee7733', '#cc3311', '#ee3377', '#bbbbbb', '#332288', '#88ccee', '#44aa99', '#117733', '#999933', '#ddcc77'],
            'Nature Style': ['#0072b2', '#d55e00', '#009e73', '#cc79a7', '#f0e442', '#56b4e9', '#e69f00', '#000000', '#52854c', '#ffdb6d', '#d16103', '#c4961a', '#293352'],
            'Science Style': ['#3b4992', '#ee0000', '#008b45', '#631879', '#008280', '#bb0021', '#5f559b', '#a20056', '#808180', '#1b1919', '#0f8b8d', '#91d1c2', '#f39b7f'],
            'IEEE Standard': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'],
            'Matplotlib Default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'],
            'Seaborn Deep': ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd', '#8dd3c7', '#ccb974', '#e1c72b', '#7f7f7f', '#e377c2'],
            'Seaborn Colorblind': ['#0072b2', '#e69f00', '#009e73', '#cc79a7', '#56b4e9', '#f0e442', '#d55e00', '#000000', '#882255', '#332288', '#117733', '#44aa99', '#999933', '#aa4499', '#ddcc77'],
            'Viridis': ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#27ae80', '#5ec962', '#a0da39', '#fde725', '#f0f921', '#e6e600', '#ffff88', '#ffea00', '#ffb000', '#ff6b00'],
            'Plasma': ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921', '#ffff88', '#ffe617', '#ffcc00', '#ffa500', '#ff6b00'],
            'Inferno': ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4', '#ffff88', '#ffe617', '#ffcc00', '#ffa500', '#ff6b00'],
            'Magma': ['#000004', '#180f3d', '#440f76', '#721f81', '#9e2f7f', '#cd4071', '#f1605d', '#fd9567', '#fec98d', '#fcfebf', '#ffff88', '#ffe617', '#ffcc00', '#ffa500', '#ff6b00'],
            'Cividis': ['#00204d', '#00336e', '#1d4e89', '#3d679e', '#5a7fb0', '#7b9bc4', '#9fb7d7', '#c4d2e9', '#e8e8e8', '#f1e8c6', '#f9e1a1', '#fcd678', '#fac228', '#f5a000', '#e67e00']
        }

        # 默认配色方案
        self.current_scheme = 'Tableau 20'
        self.available_colors = self.color_schemes[self.current_scheme].copy()
        self.color_index = 0
        
        # 初始化所有Tkinter变量
        self.smooth_enabled = tk.BooleanVar(value=False)
        self.window_length = tk.IntVar(value=5)
        self.polyorder = tk.IntVar(value=2)

        # 中文配色方案名称映射
        self.scheme_names_zh = {
            'ColorBrewer Set1': 'ColorBrewer 集合1',
            'ColorBrewer Set2': 'ColorBrewer 集合2',
            'ColorBrewer Set3': 'ColorBrewer 集合3',
            'ColorBrewer Dark2': 'ColorBrewer 深色2',
            'ColorBrewer Paired': 'ColorBrewer 配对',
            'Tableau 10': 'Tableau 10色',
            'Tableau 20': 'Tableau 20色',
            'Okabe-Ito (Colorblind)': 'Okabe-Ito (色盲友好)',
            'Wong (Colorblind)': 'Wong (色盲友好)',
            'Tol Bright': 'Tol 明亮',
            'Tol Vibrant': 'Tol 鲜艳',
            'Nature Style': 'Nature 风格',
            'Science Style': 'Science 风格',
            'IEEE Standard': 'IEEE 标准',
            'Matplotlib Default': 'Matplotlib 默认',
            'Seaborn Deep': 'Seaborn 深色',
            'Seaborn Colorblind': 'Seaborn 色盲友好',
            'Viridis': 'Viridis',
            'Plasma': 'Plasma',
            'Inferno': 'Inferno',
            'Magma': 'Magma',
            'Cividis': 'Cividis'
        }
        # 创建中文名称到英文键的反向映射
        self.scheme_names_reverse = {v: k for k, v in self.scheme_names_zh.items()}
        
        # 初始化界面组件
        self.create_widgets()
        self.init_chart()
        
    def create_widgets(self):
        # 创建控制面板
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="File Management", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Button(
            file_frame,
            text="Add JSON Files",
            command=self.add_multiple_files  # 支持多文件选择
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            file_frame,
            text="Clear All Files",
            command=self.clear_all_files
        ).pack(side=tk.LEFT, padx=2)
        
        # 坐标轴选择组件
        axis_frame = ttk.Frame(control_frame)
        axis_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(axis_frame, text="X Axis Field:").pack(side=tk.LEFT, padx=2)
        self.x_axis_combobox = ttk.Combobox(axis_frame, width=15)
        self.x_axis_combobox.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(axis_frame, text="Y Axis Field:").pack(side=tk.LEFT, padx=2)
        self.y_axis_combobox = ttk.Combobox(axis_frame, width=15)
        self.y_axis_combobox.pack(side=tk.LEFT, padx=2)
        
        # 操作按钮
        ttk.Button(
            axis_frame,
            text="Refresh Chart",
            command=self.update_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # 右侧控制面板
        control_right = ttk.Frame(control_frame)
        control_right.pack(side=tk.RIGHT)

        # 配色方案选择
        ttk.Label(control_right, text="配色方案:").pack(side=tk.LEFT, padx=(0,5))
        self.scheme_var = tk.StringVar(value=self.scheme_names_zh.get(self.current_scheme, self.current_scheme))
        self.scheme_combo = ttk.Combobox(
            control_right,
            textvariable=self.scheme_var,
            values=list(self.scheme_names_zh.values()),
            width=15,
            state='readonly'
        )
        self.scheme_combo.pack(side=tk.LEFT, padx=(0,10))
        self.scheme_combo.bind('<<ComboboxSelected>>', self.on_global_scheme_changed)

        # 保存按钮
        ttk.Button(
            control_right,
            text="Save Chart",
            command=self.save_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # 平滑功能控件
        ttk.Checkbutton(
            control_right,
            text="Smooth Curve",
            variable=self.smooth_enabled,
            command=self.toggle_smooth_options
        ).pack(side=tk.LEFT, padx=5)
        
        self.smooth_frame = ttk.Frame(control_right)
        
        ttk.Label(self.smooth_frame, text="Window:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=3,
            to=21,
            increment=2,
            textvariable=self.window_length,
            width=5
        ).pack(side=tk.LEFT)
        
        ttk.Label(self.smooth_frame, text="Order:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=2,
            to=5,
            textvariable=self.polyorder,
            width=3
        ).pack(side=tk.LEFT)
        
        # 创建文件列表和图表区域
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧文件列表
        list_frame = ttk.LabelFrame(main_frame, text="Loaded Files", padding=5)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.file_listbox = tk.Listbox(list_frame, width=30, height=20)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        
        # 文件列表按钮
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text="Rename Line",
            command=self.rename_selected_file
        ).pack(side=tk.LEFT, padx=1)

        ttk.Button(
            btn_frame,
            text="Change Color",
            command=self.change_file_color
        ).pack(side=tk.LEFT, padx=1)

        ttk.Button(
            btn_frame,
            text="Remove Selected",
            command=self.remove_selected_file
        ).pack(side=tk.LEFT, padx=1)
        
        # 图表区域
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chart_frame = chart_frame
        
    def toggle_smooth_options(self):
        """控制平滑参数控件的显示"""
        if self.smooth_enabled.get():
            self.smooth_frame.pack(side=tk.LEFT, padx=5)
        else:
            self.smooth_frame.pack_forget()

    def on_global_scheme_changed(self, event):
        """全局配色方案切换"""
        chinese_name = self.scheme_var.get()
        new_scheme = self.scheme_names_reverse.get(chinese_name, chinese_name)
        if new_scheme != self.current_scheme:
            self.current_scheme = new_scheme
            self.available_colors = self.color_schemes[new_scheme].copy()
            self.color_index = 0  # 重置颜色索引

            # 重新为所有文件分配新配色方案的颜色
            for i, filename in enumerate(self.file_data.keys()):
                if i < len(self.available_colors):
                    self.file_colors[filename] = self.available_colors[i]
                else:
                    # 如果配色方案颜色不够，循环使用
                    self.file_colors[filename] = self.available_colors[i % len(self.available_colors)]

            self.update_file_list()
            self.update_chart()
    
    def apply_smoothing(self, y_data):
        """应用Savitzky-Golay平滑"""
        if not savgol_filter:
            messagebox.showerror("Error", "Please install scipy library: pip install scipy")
            return y_data
            
        window = self.window_length.get()
        polyorder = self.polyorder.get()
        
        # 确保窗口为奇数且小于数据长度
        window = min(len(y_data)//2*2-1, window)  # 最大可用奇数
        window = max(3, window)
        if window % 2 == 0:
            window += 1
            
        try:
            return savgol_filter(y_data, window, polyorder)
        except Exception as e:
            messagebox.showerror("Smoothing Error", str(e))
            return y_data
    
    def save_chart(self):
        """保存图表到文件"""
        if not hasattr(self, 'figure') or len(self.figure.axes) == 0:
            messagebox.showwarning("Warning", "Please generate a chart first")
            return
            
        filetypes = [
            ('PNG Image', '*.png'),
            ('JPEG Image', '*.jpg'),
            ('PDF Document', '*.pdf'),
            ('SVG Vector', '*.svg')
        ]
        
        path = filedialog.asksaveasfilename(
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if path:
            try:
                self.figure.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Chart saved to: {path}")
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))
    
    def init_chart(self):
        # 初始化图表
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def add_multiple_files(self):
        """添加多个JSON文件 - 支持同时选择多个文件"""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("JSON files", "*.json")],
            title="Select JSON Files - Multiple Selection Supported"
        )
        
        if file_paths:
            loaded_count = 0
            error_files = []
            
            for file_path in file_paths:
                try:
                    if self.load_file_data(file_path):
                        filename = file_path.split('/')[-1]  # 获取文件名
                        
                        # 检查是否已经加载了该文件
                        if filename in self.file_data:
                            # 如果文件已存在，添加数字后缀
                            base_name = filename.replace('.json', '')
                            counter = 1
                            while f"{base_name}_{counter}.json" in self.file_data:
                                counter += 1
                            filename = f"{base_name}_{counter}.json"
                        
                        self.file_data[filename] = self.temp_data
                        
                        # 分配颜色（使用当前配色方案）
                        color = self.available_colors[self.color_index % len(self.available_colors)]
                        self.file_colors[filename] = color
                        self.color_index += 1
                        
                        # 设置默认标签为文件名
                        self.file_labels[filename] = filename
                        
                        loaded_count += 1
                except Exception as e:
                    error_files.append(f"{file_path}: {str(e)}")
            
            if loaded_count > 0:
                # 更新文件列表显示
                self.update_file_list()
                # 更新坐标轴选项
                self.update_axis_options()
                
                success_msg = f"{loaded_count} file(s) loaded successfully"
                if error_files:
                    success_msg += f"\n\nFailed files:\n" + "\n".join(error_files[:3])  # 最多显示3个错误
                
                messagebox.showinfo("Load Result", success_msg)
    
    def load_file_data(self, file_path):
        """加载单个文件的数据"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # 强制转换为字典格式
            if not isinstance(data, dict):
                data = {"data": [data]}
                
            # 转换所有值为列表
            processed_data = {}
            for k, v in data.items():
                if not isinstance(v, list):
                    processed_data[k] = [v]
                else:
                    processed_data[k] = v
                    
            self.temp_data = processed_data
            return True
        except Exception as e:
            return False
    
    def update_file_list(self):
        """更新文件列表显示"""
        self.file_listbox.delete(0, tk.END)
        for filename in self.file_data.keys():
            label = self.file_labels.get(filename, filename)
            color = self.file_colors.get(filename, 'black')
            self.file_listbox.insert(tk.END, f"{label} ({filename})")
            
            # 设置颜色（如果可能的话）
            try:
                index = list(self.file_data.keys()).index(filename)
                self.file_listbox.itemconfig(index, {'fg': color})
            except:
                pass
    
    def update_axis_options(self):
        """更新坐标轴选项"""
        if not self.file_data:
            return
            
        # 获取所有文件的共同字段
        all_keys = set()
        for data in self.file_data.values():
            all_keys.update(data.keys())
        
        if all_keys:
            keys = ['num'] + list(all_keys)
            self.x_axis_combobox['values'] = keys
            self.y_axis_combobox['values'] = list(all_keys)
            
            # 设置默认值
            if not self.x_axis_combobox.get():
                self.x_axis_combobox.set('num')
            if not self.y_axis_combobox.get() and all_keys:
                self.y_axis_combobox.set(next(iter(all_keys)))
    
    def rename_selected_file(self):
        """重命名选中的文件线条"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file first")
            return

        index = selection[0]
        filename = list(self.file_data.keys())[index]

        new_label = simpledialog.askstring(
            "Rename Line",
            f"Enter new line name for '{filename}':",
            initialvalue=self.file_labels.get(filename, filename)
        )

        if new_label and new_label.strip():
            self.file_labels[filename] = new_label.strip()
            self.update_file_list()
            self.update_chart()

    def change_file_color(self):
        """更改选中文件的颜色"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file first")
            return

        index = selection[0]
        filename = list(self.file_data.keys())[index]
        current_color = self.file_colors.get(filename, '#1f77b4')

        # 显示颜色选择器对话框
        color_picker = ColorPickerDialog(self.root, current_color, self.color_schemes)
        new_color, selected_scheme = color_picker.show()

        if new_color != current_color:
            self.file_colors[filename] = new_color
            if selected_scheme:
                self.current_scheme = selected_scheme
                # 更新默认颜色列表
                self.available_colors = self.color_schemes[self.current_scheme].copy()
            self.update_file_list()
            self.update_chart()

            # 显示颜色更改成功消息
            messagebox.showinfo("成功", f"'{filename}' 的颜色已更改为 {new_color}")
    
    def remove_selected_file(self):
        """移除选中的文件"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        index = selection[0]
        filename = list(self.file_data.keys())[index]
        
        if messagebox.askyesno("Confirm", f"Remove file '{filename}'?"):
            del self.file_data[filename]
            del self.file_labels[filename]
            del self.file_colors[filename]
            
            self.update_file_list()
            self.update_chart()
            
            if not self.file_data:
                self.x_axis_combobox['values'] = []
                self.y_axis_combobox['values'] = []
    
    def clear_all_files(self):
        """清空所有文件"""
        if not self.file_data:
            return

        if messagebox.askyesno("Confirm", "Clear all files?"):
            self.file_data.clear()
            self.file_labels.clear()
            self.file_colors.clear()
            self.color_index = 0

            self.file_listbox.delete(0, tk.END)
            self.x_axis_combobox['values'] = []
            self.y_axis_combobox['values'] = []

            self.update_chart()
    
    def validate_entries(self):
        """验证输入"""
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        
        if not x_key or not y_key:
            return False
            
        if not self.file_data:
            return False
            
        # 检查至少有一个文件包含所需的字段
        for data in self.file_data.values():
            if y_key in data:
                if x_key == 'num' or x_key in data:
                    return True
        
        return False
    
    def process_file_data(self, filename, data, x_key, y_key):
        """处理单个文件的数据"""
        try:
            # 获取Y轴数据并过滤无效值
            raw_y = data[y_key]
            y_data = []
            for y in raw_y:
                try:
                    y_data.append(float(y))
                except (ValueError, TypeError):
                    continue
            
            # 生成X轴数据
            if x_key == 'num':
                x_data = list(range(len(y_data)))
            else:
                x_data = []
                for x in data[x_key]:
                    try:
                        x_data.append(float(x))
                    except (ValueError, TypeError):
                        continue
            
            # 自动对齐数据长度并限制X值小于10000
            min_length = min(len(x_data), len(y_data))
            filtered_x, filtered_y = [], []
            for i in range(min_length):
                if x_data[i] < 10000:
                    filtered_x.append(x_data[i])
                    filtered_y.append(y_data[i])
            
            if not filtered_x:
                return None
            
            return filtered_x, filtered_y
        except:
            return None
    
    def update_chart(self):
        """更新图表 - 确保显示所有曲线"""
        if not self.validate_entries():
            return
            
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        lines_plotted = 0
        
        # 调试信息
        print(f"Attempting to plot {len(self.file_data)} files...")
        print(f"X axis: {x_key}, Y axis: {y_key}")
        
        # 为每个文件绘制线条
        for filename, data in self.file_data.items():
            print(f"Processing file: {filename}")
            print(f"Available fields: {list(data.keys())}")
            
            if y_key not in data:
                print(f"Skipping {filename}: missing {y_key}")
                continue
                
            processed = self.process_file_data(filename, data, x_key, y_key)
            if not processed or not all(len(d) > 0 for d in processed):
                print(f"Skipping {filename}: no valid data")
                continue
                
            x_data, y_data = processed
            print(f"Plotting {filename}: {len(x_data)} data points")
            
            # 应用平滑
            if self.smooth_enabled.get():
                y_data = self.apply_smoothing(y_data)
            
            # 获取线条标签和颜色
            line_label = self.file_labels.get(filename, filename)
            color = self.file_colors.get(filename, 'steelblue')
            
            # 绘制线条
            ax.plot(
                x_data, y_data,
                linestyle='-',
                color=color,
                linewidth=1,
                label=line_label,
                marker='None',
                markersize=4,
                alpha=0.8
            )
            
            lines_plotted += 1
        
        print(f"Total lines plotted: {lines_plotted}")
        
        if lines_plotted > 0:
            # 使用英文标题，去除空白字符
            chart_title = f"Multi-File Data Comparison - {y_key}"
            ax.set_title(chart_title, fontsize=14, pad=15)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            
            # 使用英文标签
            x_label = x_key if x_key != 'num' else 'Data Point Index'
            ax.set_xlabel(x_label, fontsize=12, labelpad=10)
            ax.set_ylabel(y_key, fontsize=12, labelpad=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            messagebox.showwarning("Warning", "No data to plot")


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiCurveChartApp(root)
    root.mainloop()