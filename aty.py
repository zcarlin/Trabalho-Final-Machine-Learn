import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import hashlib
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, auc
from itertools import combinations, cycle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import random
import threading
import sys
from io import StringIO
import requests
from io import BytesIO

# Configuração do tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def show_login_screen():
    # Tela de login
    login_window = tk.Tk()
    login_window.title("Login")
    login_window.geometry("600x700")
    login_window.configure(bg='#2196F3')  # Azul predominante
    
    # Centraliza a janela
    screen_width = login_window.winfo_screenwidth()
    screen_height = login_window.winfo_screenheight()
    x = (screen_width/2) - (600/2)
    y = (screen_height/2) - (700/2)
    login_window.geometry(f"600x700+{int(x)}+{int(y)}")
    
    # Frame principal
    main_frame = tk.Frame(login_window, bg='#2196F3')  # Fundo azul
    main_frame.pack(pady=30, fill="both", expand=True)
    
    # URLs das imagens
    sonic_url = "https://raw.githubusercontent.com/zcarlin/Trabalho-Final-Machine-Learn/main/sonic.png"
    citha_url = "https://raw.githubusercontent.com/zcarlin/Trabalho-Final-Machine-Learn/main/citha.png"

    # Carrega e redimensiona as imagens
    try:
        # Carrega imagem do Sonic
        response = requests.get(sonic_url, timeout=10)
        if response.status_code == 200:
            sonic_image = Image.open(BytesIO(response.content))
            sonic_photo = ImageTk.PhotoImage(sonic_image.resize((150, 150)))
        else:
            sonic_photo = None
        
        # Carrega imagem da CITHA
        response = requests.get(citha_url, timeout=10)
        if response.status_code == 200:
            citha_image = Image.open(BytesIO(response.content))
            citha_photo = ImageTk.PhotoImage(citha_image.resize((150, 150)))
        else:
            citha_photo = None
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao carregar imagens: {str(e)}")
        sonic_photo = None
        citha_photo = None

    # Layout
    tk.Label(main_frame, text="Sistema de Análise de Safra", 
             font=('Arial', 24, 'bold'), bg='#2196F3', fg='white').pack(pady=30)
    
    # Frame para as imagens
    image_frame = tk.Frame(main_frame, bg='#2196F3')
    image_frame.pack(pady=20)
    
    if sonic_photo and citha_photo:
        # Frame para o lado esquerdo (Sonic)
        left_frame = tk.Frame(image_frame, bg='#2196F3')
        left_frame.pack(side='left', padx=20)
        
        # Frame para o lado direito (Citha)
        right_frame = tk.Frame(image_frame, bg='#2196F3')
        right_frame.pack(side='right', padx=20)
        
        # Labels com texto
        tk.Label(left_frame, text="Crias Da Python", font=('Arial', 12, 'bold'), 
                 bg='#2196F3', fg='white').pack(pady=(0, 5))
        tk.Label(right_frame, text="IFAM-CITHA", font=('Arial', 12, 'bold'), 
                 bg='#2196F3', fg='white').pack(pady=(0, 5))
        
        # Labels com imagens
        tk.Label(left_frame, image=sonic_photo, bg='#2196F3').pack()
        tk.Label(right_frame, image=citha_photo, bg='#2196F3').pack()
    
    # Campos de entrada
    username_var = tk.StringVar()
    password_var = tk.StringVar()
    
    tk.Label(main_frame, text="Usuário:", font=('Arial', 14), bg='#2196F3', fg='white').pack(pady=(30, 10))
    username_entry = tk.Entry(main_frame, textvariable=username_var, font=('Arial', 14), bg='white')
    username_entry.pack(pady=10, padx=40, fill="x")
    
    tk.Label(main_frame, text="Senha:", font=('Arial', 14), bg='#2196F3', fg='white').pack(pady=10)
    password_entry = tk.Entry(main_frame, textvariable=password_var, show="*", font=('Arial', 14), bg='white')
    password_entry.pack(pady=10, padx=40, fill="x")
    
    def login(username, password):
        if username == "Walter Claudino da Silva Júnior" and password == "senha":
            # Mantém as referências para as imagens antes de destruir a janela
            main_frame.image1 = sonic_photo
            main_frame.image2 = citha_photo
            
            login_window.destroy()
            app = SafraAnalysisGUI()
            app.run()
        else:
            messagebox.showerror("Erro", "Usuário ou senha inválidos!")
            password_var.set("")
            password_entry.focus()

    # Botão de login
    login_button = tk.Button(main_frame, text="Login", font=('Arial', 14, 'bold'), 
                            bg='#4CAF50', fg='white',  # Verde
                            command=lambda: login(username_var.get(), password_var.get()))
    login_button.pack(pady=40, padx=40, fill="x")

    # Inicia o loop principal da janela de login
    login_window.mainloop()

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip, 
            text=self.text, 
            justify=tk.LEFT,
            background="#ffffe0", 
            relief=tk.SOLID, 
            borderwidth=1,
            font=("tahoma", "8", "normal")
        )
        label.pack()

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class SafraAnalysisGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Sistema de Análise de Safra com Aprendizado por Reforço")
        self.root.geometry("1400x800")
        
        # Variáveis globais do sistema original
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_pca = None
        self.df_PCA = None
        self.scaler = None
        self.pca_model = None
        self.modelo_linear = None
        self.modelo_ridge = None
        self.modelo_pca = None
        self.modelo_pca_ridge = None
        self.X_class = None
        self.y_class = None
        self.X_class_scaled = None
        self.X_class_pca = None
        self.modelo_classico = None
        self.modelo_pca_class = None
        self.preprocessador = None
        self.colunas_numericas = None
        self.colunas_binarias = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_padronizado = None
        self.pca_full = None
        self.pca = None
        self.pipeline_original = None
        self.pipeline_ridge = None
        self.X_pca_train = None
        self.X_pca_test = None
        self.y_pca_train = None
        self.y_pca_test = None
        self.y_pred_orig = None
        self.y_pred_ridge = None
        self.y_pred_pca = None
        self.y_pred_pca_ridge = None
        self.scaler_class = None
        self.pca_class = None
        self.X_train_class = None
        self.X_test_class = None
        self.y_train_class = None
        self.y_test_class = None
        self.X_train_pca = None
        self.X_test_pca = None
        
        # Controle de gráficos
        self.plots = []
        self.current_plot_index = -1
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal com grid
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configurar grid
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Painel lateral esquerdo
        self.setup_sidebar()
        
        # Área principal direita
        self.setup_main_area()
        
    def setup_sidebar(self):
        # Frame do sidebar com scrollbar
        self.sidebar_container = ctk.CTkFrame(self.main_frame)
        self.sidebar_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Canvas para scrollbar
        self.sidebar_canvas = tk.Canvas(self.sidebar_container, bg="#2b2b2b")  # Cor escura padrão do customtkinter
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        
        # Scrollbar
        self.sidebar_scrollbar = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_scrollbar.pack(side="right", fill="y")
        
        # Configurar canvas
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        
        # Frame do sidebar que será scrollável
        self.sidebar = ctk.CTkFrame(self.sidebar_canvas, width=300)
        self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        
        # Configurar scroll
        def configure_scroll(event):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
            self.sidebar_canvas.itemconfig(self.sidebar_canvas.find_withtag("all")[0], width=event.width)
        
        self.sidebar.bind("<Configure>", configure_scroll)
        self.sidebar_canvas.bind("<Configure>", configure_scroll)
        
        # Título
        title_label = ctk.CTkLabel(
            self.sidebar, 
            text="Sistema de Análise de Safra",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Seção de Dados
        data_frame = ctk.CTkFrame(self.sidebar)
        data_frame.pack(fill="x", padx=10, pady=10)
        
        data_label = ctk.CTkLabel(data_frame, text="Dados", font=ctk.CTkFont(size=16, weight="bold"))
        data_label.pack(pady=5)
        
        self.load_data_btn = ctk.CTkButton(
            data_frame, 
            text="Carregar Dados", 
            command=self.load_data_thread,
            height=35
        )
        self.load_data_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.load_data_btn, "Carrega o arquivo de dados CSV para análise")
        
        self.create_vars_btn = ctk.CTkButton(
            data_frame, 
            text="Criar Variáveis Adicionais", 
            command=self.create_additional_vars,
            state="disabled",
            height=35
        )
        self.create_vars_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.create_vars_btn, "Cria variáveis derivadas e transformações dos dados")
        
        # Seção de EDA
        eda_frame = ctk.CTkFrame(self.sidebar)
        eda_frame.pack(fill="x", padx=10, pady=10)
        
        eda_label = ctk.CTkLabel(eda_frame, text="Análise Exploratória", font=ctk.CTkFont(size=16, weight="bold"))
        eda_label.pack(pady=5)
        
        self.eda_options = ctk.CTkComboBox(
            eda_frame,
            values=[
                "Boxplot: ENSO × Produtividade",
                "Scatter: Temperatura × Produtividade",
                "Histograma de Variáveis",
                "Heatmap de Correlação",
                "Pairplot",
                "Scree Plot (PCA)",
                "Visualização PCA 2D",
                "Função de Custo 1D",
                "Função de Custo 2D",
                "Função de Custo 2D (PCA)",
                "Gráficos de Resíduos",
                "Fronteiras de Decisão (PCA)"
            ],
            state="readonly",
            width=250
        )
        self.eda_options.pack(padx=10, pady=5)
        ToolTip(self.eda_options, "Selecione o tipo de visualização exploratória desejada")
        
        self.eda_btn = ctk.CTkButton(
            eda_frame, 
            text="Visualizar", 
            command=self.run_eda,
            state="disabled",
            height=35
        )
        self.eda_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.eda_btn, "Gera a visualização selecionada")
        
        # Seção de Modelos
        models_frame = ctk.CTkFrame(self.sidebar)
        models_frame.pack(fill="x", padx=10, pady=10)
        
        models_label = ctk.CTkLabel(models_frame, text="Modelos", font=ctk.CTkFont(size=16, weight="bold"))
        models_label.pack(pady=5)
        
        self.prep_regression_btn = ctk.CTkButton(
            models_frame, 
            text="Preparar Regressão", 
            command=self.prepare_regression,
            state="disabled",
            height=35
        )
        self.prep_regression_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.prep_regression_btn, "Prepara os dados para modelos de regressão")
        
        self.train_regression_btn = ctk.CTkButton(
            models_frame, 
            text="Treinar Regressão", 
            command=self.train_regression,
            state="disabled",
            height=35
        )
        self.train_regression_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.train_regression_btn, "Treina os modelos de regressão")
        
        self.prep_classification_btn = ctk.CTkButton(
            models_frame, 
            text="Preparar Classificação", 
            command=self.prepare_classification,
            state="disabled",
            height=35
        )
        self.prep_classification_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.prep_classification_btn, "Prepara os dados para modelos de classificação")
        
        self.train_classification_btn = ctk.CTkButton(
            models_frame, 
            text="Treinar Classificação", 
            command=self.train_classification,
            state="disabled",
            height=35
        )
        self.train_classification_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.train_classification_btn, "Treina os modelos de classificação")
        
        # Seção de Aprendizado por Reforço
        rl_frame = ctk.CTkFrame(self.sidebar)
        rl_frame.pack(fill="x", padx=10, pady=10)
        
        rl_label = ctk.CTkLabel(rl_frame, text="Aprendizado por Reforço", font=ctk.CTkFont(size=16, weight="bold"))
        rl_label.pack(pady=5)
        
        self.rl_mode = ctk.CTkComboBox(
            rl_frame,
            values=["padrão", "extremo"],
            state="readonly",
            width=250
        )
        self.rl_mode.set("padrão")
        self.rl_mode.pack(padx=10, pady=5)
        ToolTip(self.rl_mode, "Selecione o modo de recompensa para o Q-learning")

        # Adiciona combobox para seleção de episódios
        self.rl_episodes = ctk.CTkComboBox(
            rl_frame,
            values=["51", "201", "999"],
            state="readonly",
            width=250
        )
        self.rl_episodes.set("201")
        self.rl_episodes.pack(padx=10, pady=5)
        ToolTip(self.rl_episodes, "Selecione o número de episódios para treinamento")

        # Adiciona combobox para seleção do epsilon
        epsilon_label = ctk.CTkLabel(rl_frame, text="Taxa de Exploração (epsilon):", font=ctk.CTkFont(size=12))
        epsilon_label.pack(padx=10, pady=(10,0))
        
        self.rl_epsilon = ctk.CTkComboBox(
            rl_frame,
            values=["0.0", "0.1", "0.5", "1.0"],
            state="readonly",
            width=250
        )
        self.rl_epsilon.set("0.9")
        self.rl_epsilon.pack(padx=10, pady=5)
        ToolTip(self.rl_epsilon, "Selecione a taxa de exploração para o Q-learning")
        
        self.rl_btn = ctk.CTkButton(
            rl_frame, 
            text="Executar Q-Learning", 
            command=self.run_qlearning,
            height=35
        )
        self.rl_btn.pack(fill="x", padx=10, pady=5)
        ToolTip(self.rl_btn, "Executa o algoritmo de Q-learning com os parâmetros selecionados")
        
        # Botão Pipeline Completo
        self.pipeline_btn = ctk.CTkButton(
            self.sidebar, 
            text="Executar Pipeline Completo", 
            command=self.run_complete_pipeline,
            state="disabled",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.pipeline_btn.pack(fill="x", padx=10, pady=20)
        ToolTip(self.pipeline_btn, "Executa todo o pipeline de análise automaticamente")
        
    def setup_main_area(self):
        # Notebook para abas
        self.notebook = ctk.CTkTabview(self.main_frame)
        self.notebook.grid(row=0, column=1, sticky="nsew")
        
        # Aba de Console
        self.console_tab = self.notebook.add("Console")
        self.setup_console_tab()
        
        # Aba de Visualizações
        self.viz_tab = self.notebook.add("Visualizações")
        self.setup_viz_tab()
        
        # Aba de Resultados
        self.results_tab = self.notebook.add("Resultados")
        self.setup_results_tab()
        
        # Aba de Dados
        self.data_tab = self.notebook.add("Dados")
        self.setup_data_tab()
        
    def setup_console_tab(self):
        # Frame para console
        console_frame = ctk.CTkFrame(self.console_tab)
        console_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Label
        console_label = ctk.CTkLabel(console_frame, text="Console de Saída", font=ctk.CTkFont(size=14, weight="bold"))
        console_label.pack(pady=5)
        
        # Text widget para console
        self.console_text = ctk.CTkTextbox(console_frame, height=600, font=("Consolas", 12))
        self.console_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Botão para limpar console
        clear_btn = ctk.CTkButton(console_frame, text="Limpar Console", command=self.clear_console)
        clear_btn.pack(pady=5)
        
    def setup_viz_tab(self):
        # Frame para visualizações
        self.viz_frame = ctk.CTkFrame(self.viz_tab)
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Label
        viz_label = ctk.CTkLabel(self.viz_frame, text="Área de Visualizações", font=ctk.CTkFont(size=14, weight="bold"))
        viz_label.pack(pady=5)
        
        # Container para plot com scrollbar
        self.plot_container = ctk.CTkFrame(self.viz_frame)
        self.plot_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Canvas para scrollbar
        self.plot_canvas = tk.Canvas(self.plot_container, bg="#2b2b2b")
        self.plot_canvas.pack(side="left", fill="both", expand=True)
        
        # Scrollbar vertical
        self.plot_scrollbar_y = ttk.Scrollbar(self.plot_container, orient="vertical", command=self.plot_canvas.yview)
        self.plot_scrollbar_y.pack(side="right", fill="y")
        
        # Scrollbar horizontal
        self.plot_scrollbar_x = ttk.Scrollbar(self.plot_container, orient="horizontal", command=self.plot_canvas.xview)
        self.plot_scrollbar_x.pack(side="bottom", fill="x")
        
        # Configurar canvas
        self.plot_canvas.configure(
            yscrollcommand=self.plot_scrollbar_y.set,
            xscrollcommand=self.plot_scrollbar_x.set
        )
        
        # Frame para matplotlib
        self.plot_frame = ctk.CTkFrame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")
        
        # Configurar scroll
        def configure_scroll(event):
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
            self.plot_canvas.itemconfig(self.plot_canvas.find_withtag("all")[0], width=event.width)
        
        self.plot_frame.bind("<Configure>", configure_scroll)
        self.plot_canvas.bind("<Configure>", configure_scroll)
        
        # Frame para botões de navegação dos gráficos
        self.nav_frame = ctk.CTkFrame(self.viz_frame)
        self.nav_frame.pack(fill="x", padx=10, pady=(0,10))
        
        self.prev_btn = ctk.CTkButton(self.nav_frame, text="Anterior", command=self.show_prev_plot, state="disabled")
        self.prev_btn.pack(side="left", padx=5)
        
        self.next_btn = ctk.CTkButton(self.nav_frame, text="Próximo", command=self.show_next_plot, state="disabled")
        self.next_btn.pack(side="left", padx=5)
        
        self.clear_plots_btn = ctk.CTkButton(self.nav_frame, text="Limpar Gráficos", command=self.clear_plots)
        self.clear_plots_btn.pack(side="right", padx=5)

    def show_prev_plot(self):
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self.display_current_plot()

    def show_next_plot(self):
        if self.current_plot_index < len(self.plots) - 1:
            self.current_plot_index += 1
            self.display_current_plot()

    def update_nav_buttons(self):
        self.prev_btn.configure(state="normal" if self.current_plot_index > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_plot_index < len(self.plots) - 1 else "disabled")
        
    def setup_results_tab(self):
        # Frame para resultados
        results_frame = ctk.CTkFrame(self.results_tab)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Label
        results_label = ctk.CTkLabel(results_frame, text="Resultados dos Modelos", font=ctk.CTkFont(size=14, weight="bold"))
        results_label.pack(pady=5)
        
        # Text widget para resultados
        self.results_text = ctk.CTkTextbox(results_frame, height=600)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
    def setup_data_tab(self):
        # Frame para dados
        data_frame = ctk.CTkFrame(self.data_tab)
        data_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Label
        data_label = ctk.CTkLabel(data_frame, text="Visualização dos Dados", font=ctk.CTkFont(size=14, weight="bold"))
        data_label.pack(pady=5)
        
        # Frame para treeview
        tree_frame = ctk.CTkFrame(data_frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side="right", fill="y")
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree_scroll_x.pack(side="bottom", fill="x")
        
        # Treeview
        self.data_tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        self.data_tree.pack(fill="both", expand=True)
        
        tree_scroll_y.config(command=self.data_tree.yview)
        tree_scroll_x.config(command=self.data_tree.xview)
        
    def log_to_console(self, message):
        """Adiciona mensagem ao console de forma thread-safe"""
        def append_message():
            formatted_message = message.replace("    ", "\t")
            self.console_text.insert("end", formatted_message + "\n")
            self.console_text.see("end")
        self.root.after(0, append_message)
        
    def clear_console(self):
        """Limpa o console"""
        self.console_text.delete("1.0", "end")
        
    def clear_plot(self):
        """Limpa a área de plot"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
    def clear_plots(self):
        """Limpa todos os gráficos armazenados"""
        self.plots = []
        self.current_plot_index = -1
        self.clear_plot()
        self.update_nav_buttons()
        self.log_to_console("Gráficos limpos com sucesso!")
            
    def show_plot(self, fig):
        """Mostra um plot matplotlib na GUI e armazena na lista"""
        # Adiciona à lista de plots
        self.plots.append(fig)
        self.current_plot_index = len(self.plots) - 1
        
        # Mostra o plot atual
        self.display_current_plot()
        
        # Muda para aba de visualizações
        self.notebook.set("Visualizações")
        
    def display_current_plot(self):
        """Exibe o plot atual da lista"""
        self.clear_plot()
        
        if 0 <= self.current_plot_index < len(self.plots):
            # Criar um frame para o canvas do matplotlib
            plot_container = ctk.CTkFrame(self.plot_frame)
            plot_container.pack(fill="both", expand=True, pady=10)
            
            # Criar o canvas do matplotlib
            canvas = FigureCanvasTkAgg(self.plots[self.current_plot_index], master=plot_container)
            canvas.draw()
            
            # Adicionar o canvas ao frame
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Adicionar barra de ferramentas de navegação do matplotlib
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, plot_container)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Atualiza estado dos botões de navegação
        self.update_nav_buttons()
            
    def update_data_view(self):
        """Atualiza a visualização dos dados"""
        if self.df is None:
            return
            
        # Limpa treeview
        self.data_tree.delete(*self.data_tree.get_children())
        
        # Configura colunas
        self.data_tree['columns'] = list(self.df.columns)
        self.data_tree['show'] = 'tree headings'
        
        # Configura cabeçalhos
        self.data_tree.heading("#0", text="Índice")
        self.data_tree.column("#0", width=100)
        
        for col in self.df.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
            
        # Adiciona dados
        for idx, row in self.df.iterrows():
            values = [str(val) for val in row.values]
            self.data_tree.insert("", "end", text=str(idx), values=values)
            
    def load_data_thread(self):
        """Carrega dados em thread separada"""
        threading.Thread(target=self.load_data, daemon=True).start()
        
    def load_data(self):
        """Carrega e prepara os dados iniciais"""
        try:
            self.log_to_console("Carregando dados...")
            
            # Abrindo o Arquivo
            self.df = pd.read_csv("DataSet.csv", sep=';', decimal=',')

            # Renomeando Colunas
            self.df.rename(columns={ 
                'chuva_durante_floração_mm': 'chuva_flor', 
                'chuva_durante_colheita_mm': 'chuva_colheita', 
                'chuva_total_anual_mm': 'chuva_total', 
                'anomalia_chuva_floração_mm': 'anomalia_flor',
                'temperatura_média_floração_C': 'temp_flor', 
                'umidade_relativa_média_floração_%': 'umid_flor', 
                'evento_ENSO': 'ENSO', 
                'produtividade_kg_por_ha': 'produtividade', 
                'produtividade_safra': 'safra' 
            }, inplace=True) 

            # Transformando em escala fracionaria 
            self.df['umid_flor'] = self.df['umid_flor'] / 100 
            self.df.set_index('ano', inplace=True) 
            
            self.log_to_console("Dados carregados com sucesso!")
            self.log_to_console(f"\nForma do DataFrame: {self.df.shape}")
            self.log_to_console(f"Colunas: {list(self.df.columns)}")
            
            # Captura informações do DataFrame
            buffer = StringIO()
            self.df.info(buf=buffer)
            self.log_to_console("\nInformações do DataFrame:")
            self.log_to_console(buffer.getvalue())
            
            # Valores ausentes
            self.log_to_console("\nValores ausentes por coluna:")
            self.log_to_console(str(self.df.isnull().sum()))
            
            # Resumo estatístico
            self.log_to_console("\nResumo estatístico:")
            self.log_to_console(str(self.df.describe().T))
            
            # Atualiza visualização dos dados
            self.root.after(0, self.update_data_view)
            
            # Habilita botões
            self.root.after(0, self.create_vars_btn.configure(state="normal"))
            self.root.after(0, self.eda_btn.configure(state="normal"))
            self.root.after(0, self.pipeline_btn.configure(state="normal"))
            
        except Exception as e:
            self.log_to_console(f"Erro ao carregar dados: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao carregar dados: {str(e)}")
            
    def create_additional_vars(self):
        """Cria variáveis sazonais e climáticas adicionais"""
        try:
            self.log_to_console("\nCriando variáveis adicionais...")
            
            # 1. Chuva relativa durante floração 
            self.df['chuva_relativa'] = self.df['chuva_flor'] / self.df['chuva_total'] 
            # 2. Binário: anomalia positiva ou não 
            self.df['anomalia_bin'] = (self.df['anomalia_flor'] > 0).astype(int) 
            # 3. Codificar ENSO como variáveis dummies 
            self.df = pd.get_dummies(self.df, columns=['ENSO'], drop_first=True)
            
            self.log_to_console("Variáveis adicionais criadas!")
            self.log_to_console(f"Novas colunas: {list(self.df.columns)}")
            
            # Atualiza visualização
            self.update_data_view()
            
            # Habilita botões de preparação
            self.prep_regression_btn.configure(state="normal")
            self.prep_classification_btn.configure(state="normal")
            
        except Exception as e:
            self.log_to_console(f"Erro ao criar variáveis: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao criar variáveis: {str(e)}")
            
    def run_eda(self):
        """Executa análise exploratória selecionada"""
        try:
            option = self.eda_options.get()
            
            if option == "Boxplot: ENSO × Produtividade":
                self.boxplot_enso()
            elif option == "Scatter: Temperatura × Produtividade":
                self.scatterplot_temp()
            elif option == "Histograma de Variáveis":
                self.histogram_vars()
            elif option == "Heatmap de Correlação":
                self.heatmap_correlation()
            elif option == "Pairplot":
                self.pairplot_vars()
            elif option == "Scree Plot (PCA)":
                self.scree_plot()
            elif option == "Visualização PCA 2D":
                self.plot_pca_2d()
            elif option == "Função de Custo 1D":
                self.plot_cost_function_1d()
            elif option == "Função de Custo 2D":
                self.plot_cost_function_2d()
            elif option == "Função de Custo 2D (PCA)":
                self.plot_cost_function_2d_pca()
            elif option == "Gráficos de Resíduos":
                self.plot_residuals()
            elif option == "Fronteiras de Decisão (PCA)":
                self.plot_decision_boundaries()
                
        except Exception as e:
            self.log_to_console(f"Erro na análise exploratória: {str(e)}")
            messagebox.showerror("Erro", f"Erro na análise exploratória: {str(e)}")
            
    def boxplot_enso(self):
        """Cria boxplot ENSO × Produtividade"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Recria a coluna ENSO temporariamente para o plot
            df_temp = self.df.copy()
            
            # Identifica o estado ENSO baseado nas colunas dummy
            if 'ENSO_La Niña' in df_temp.columns and 'ENSO_Neutro' in df_temp.columns:
                df_temp['ENSO'] = 'El Niño'  # Default
                df_temp.loc[df_temp['ENSO_La Niña'] == 1, 'ENSO'] = 'La Niña'
                df_temp.loc[df_temp['ENSO_Neutro'] == 1, 'ENSO'] = 'Neutro'
            else:
                self.log_to_console("Colunas ENSO não encontradas. Execute 'Criar Variáveis Adicionais' primeiro.")
                return
            
            sns.boxplot(
                data=df_temp,
                x='ENSO',
                y='produtividade',
                order=['La Niña', 'Neutro', 'El Niño'],
                ax=ax
            )
            ax.set_title('Produtividade vs. Evento ENSO', fontsize=14)
            ax.set_xlabel('Evento ENSO', fontsize=12)
            ax.set_ylabel('Produtividade (kg/ha)', fontsize=12)
            
            plt.tight_layout()
            self.show_plot(fig)
            return fig
            
        except Exception as e:
            self.log_to_console(f"Erro ao criar boxplot: {str(e)}")
            return None
        
    def scatterplot_temp(self):
        """Cria scatterplot Temperatura × Produtividade"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Recria a coluna ENSO temporariamente
            df_temp = self.df.copy()
            if 'ENSO_La Niña' in df_temp.columns and 'ENSO_Neutro' in df_temp.columns:
                df_temp['ENSO'] = 'El Niño'
                df_temp.loc[df_temp['ENSO_La Niña'] == 1, 'ENSO'] = 'La Niña'
                df_temp.loc[df_temp['ENSO_Neutro'] == 1, 'ENSO'] = 'Neutro'
            
            sns.scatterplot(
                data=df_temp,
                x='temp_flor',
                y='produtividade',
                hue='ENSO' if 'ENSO' in df_temp.columns else None,
                s=80,
                alpha=0.8,
                ax=ax
            )
            ax.set_title('Temperatura durante floração vs. Produtividade', fontsize=14)
            ax.set_xlabel('Temperatura média durante floração (°C)', fontsize=12)
            ax.set_ylabel('Produtividade (kg/ha)', fontsize=12)
            
            plt.tight_layout()
            self.show_plot(fig)
            return fig
            
        except Exception as e:
            self.log_to_console(f"Erro ao criar scatterplot: {str(e)}")
            return None
        
    def histogram_vars(self):
        """Cria histogramas das variáveis numéricas"""
        try:
            numeric_cols = self.df.select_dtypes(include='number')
            n_cols = len(numeric_cols.columns)
            n_rows = int(np.ceil(n_cols / 3))
            
            fig, axes = plt.subplots(n_rows, 3, figsize=(8, 2.5*n_rows))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols.columns):
                if i < len(axes):
                    numeric_cols[col].hist(bins=15, ax=axes[i])
                    axes[i].set_title(col, fontsize=8)
                    axes[i].tick_params(axis='both', which='major', labelsize=7)
                    
            # Remove eixos extras
            for i in range(len(numeric_cols.columns), len(axes)):
                fig.delaxes(axes[i])
                
            plt.suptitle("Distribuições das variáveis numéricas", fontsize=10)
            plt.tight_layout()
            self.show_plot(fig)
            return fig
            
        except Exception as e:
            self.log_to_console(f"Erro ao criar histogramas: {str(e)}")
            return None
        
    def heatmap_correlation(self):
        """Cria heatmap de correlação"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Seleciona só as colunas numéricas
            variaveis_numericas = self.df.select_dtypes(include='number')
            
            # Calcula a matriz de correlação
            correlacao = variaveis_numericas.corr()
            
            # Heatmap
            sns.heatmap(
                correlacao,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                linewidths=0.5,
                square=True,
                cbar_kws={"shrink": .8},
                vmin=-1, vmax=1,
                ax=ax
            )
            ax.set_title('Matriz de Correlação entre Variáveis Numéricas')
            
            plt.tight_layout()
            self.show_plot(fig)
            return fig
            
        except Exception as e:
            self.log_to_console(f"Erro ao criar heatmap: {str(e)}")
            return None
        
    def pairplot_vars(self):
        """Cria pairplot das variáveis"""
        try:
            self.log_to_console("Gerando pairplot... Isso pode demorar um pouco.")
            
            # Seleciona as variáveis numéricas principais
            cols_plot = ['chuva_flor', 'chuva_colheita', 'chuva_total',
                        'anomalia_flor', 'temp_flor', 'umid_flor', 'produtividade']
            
            # Verifica se as colunas existem
            cols_plot = [col for col in cols_plot if col in self.df.columns]
            
            # Cria o pairplot com tamanho reduzido
            g = sns.pairplot(
                self.df[cols_plot],
                corner=True,
                diag_kind='hist',
                plot_kws={'alpha': 0.7, 's': 20, 'edgecolor': 'k'},
                height=1.5,  # Altura de cada subplot
                aspect=1.2    # Proporção largura/altura
            )
            
            # Ajusta o tamanho da fonte dos títulos
            for ax in g.axes.flat:
                if ax is not None:
                    ax.set_title(ax.get_title(), fontsize=8)
                    ax.tick_params(axis='both', which='major', labelsize=7)
            
            g.fig.suptitle("Matriz de Dispersão entre Variáveis", fontsize=10, y=1.02)
            
            # Ajusta o layout para evitar sobreposição
            plt.tight_layout()
            
            self.show_plot(g.fig)
            return g.fig
            
        except Exception as e:
            self.log_to_console(f"Erro ao criar pairplot: {str(e)}")
            return None
            
    def prepare_regression(self):
        """Prepara dados para regressão"""
        try:
            self.log_to_console("\n=== PREPARANDO DADOS PARA REGRESSÃO ===")
            
            # 1. Definindo X e y
            self.X = self.df.drop(columns=['produtividade', 'safra'])
            self.y = self.df['produtividade']
            
            # 2. Verificando colunas numéricas e binárias
            self.colunas_numericas = ['chuva_flor', 'chuva_colheita', 'chuva_total',
                                     'anomalia_flor', 'temp_flor', 'umid_flor', 'chuva_relativa']
            self.colunas_binarias = ['anomalia_bin', 'ENSO_La Niña', 'ENSO_Neutro']
            
            # 3. Criando o transformador
            self.preprocessador = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.colunas_numericas),
                    ('bin', 'passthrough', self.colunas_binarias)
                ]
            )
            
            # 4. Separando treino e teste
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, shuffle=False
            )
            
            # 5. Aplicando pré-processamento
            self.X_scaled = self.preprocessador.fit_transform(self.X)
            self.X_train_scaled = self.preprocessador.transform(self.X_train)
            self.X_test_scaled = self.preprocessador.transform(self.X_test)
            
            # 6. PCA para análise completa (Scree Plot)
            self.pca_full = PCA()
            self.pca_full.fit(self.X_scaled)
            
            # 7. PCA para redução de dimensionalidade
            self.pca_model = PCA(n_components=2)
            self.X_pca = self.pca_model.fit_transform(self.X_scaled)
            self.X_pca_train = self.pca_model.transform(self.X_train_scaled)
            self.X_pca_test = self.pca_model.transform(self.X_test_scaled)
            
            # 8. Criando DataFrame PCA
            self.df_PCA = pd.DataFrame(self.X_pca, columns=['PC1', 'PC2'], index=self.X.index)
            self.df_PCA['produtividade'] = self.y
            self.df_PCA['safra'] = self.df['safra']
            
            # 9. Inicializando o scaler para comparação de modelos
            self.scaler = StandardScaler()
            self.scaler.fit(self.X[self.colunas_numericas])
            
            # 10. Criando DataFrame com features transformadas
            self.X_scaled_df = pd.DataFrame(
                self.X_scaled,
                columns=self.colunas_numericas + self.colunas_binarias,
                index=self.X.index
            )
            
            self.log_to_console(f"Dados preparados para regressão!")
            self.log_to_console(f"Tamanho do conjunto de treino: {len(self.X_train)}")
            self.log_to_console(f"Tamanho do conjunto de teste: {len(self.X_test)}")
            
            # Habilita botão de treino
            self.train_regression_btn.configure(state="normal")
            
        except Exception as e:
            self.log_to_console(f"Erro ao preparar regressão: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao preparar regressão: {str(e)}")

    def train_regression(self):
        """Treina modelos de regressão"""
        try:
            self.log_to_console("\n=== TREINANDO MODELOS DE REGRESSÃO ===")

            # 1. Modelo Linear
            self.modelo_linear = LinearRegression()
            self.modelo_linear.fit(self.X_train_scaled, self.y_train)
            self.y_pred_linear = self.modelo_linear.predict(self.X_test_scaled)
            mse_linear = mean_squared_error(self.y_test, self.y_pred_linear)
            rmse_linear = mse_linear ** 0.5
            r2_linear = r2_score(self.y_test, self.y_pred_linear)
            self.log_to_console(f"[Regressão linear] RMSE: {rmse_linear:.2f} | R²: {r2_linear:.2%}")

            # 2. Modelo Ridge
            lambda_regressao = 1
            self.modelo_ridge = Ridge(alpha=lambda_regressao)
            self.modelo_ridge.fit(self.X_train_scaled, self.y_train)
            self.y_pred_ridge = self.modelo_ridge.predict(self.X_test_scaled)
            mse_ridge = mean_squared_error(self.y_test, self.y_pred_ridge)
            rmse_ridge = mse_ridge ** 0.5
            r2_ridge = r2_score(self.y_test, self.y_pred_ridge)
            self.log_to_console(f"[Ridge (λ={lambda_regressao})] RMSE: {rmse_ridge:.2f} | R²: {r2_ridge:.2%}")

            # 3. Modelo PCA Linear
            self.modelo_pca = LinearRegression()
            self.modelo_pca.fit(self.X_pca_train, self.y_train)
            self.y_pred_pca = self.modelo_pca.predict(self.X_pca_test)
            mse_pca = mean_squared_error(self.y_test, self.y_pred_pca)
            rmse_pca = mse_pca ** 0.5
            r2_pca = r2_score(self.y_test, self.y_pred_pca)
            self.log_to_console(f"[PCA + Regressão linear] RMSE: {rmse_pca:.2f} | R²: {r2_pca:.2%}")

            # 4. Modelo PCA Ridge
            self.modelo_pca_ridge = Ridge(alpha=lambda_regressao)
            self.modelo_pca_ridge.fit(self.X_pca_train, self.y_train)
            self.y_pred_pca_ridge = self.modelo_pca_ridge.predict(self.X_pca_test)
            mse_pca_ridge = mean_squared_error(self.y_test, self.y_pred_pca_ridge)
            rmse_pca_ridge = mse_pca_ridge ** 0.5
            r2_pca_ridge = r2_score(self.y_test, self.y_pred_pca_ridge)
            self.log_to_console(f"[PCA + Ridge (λ={lambda_regressao})] RMSE: {rmse_pca_ridge:.2f} | R²: {r2_pca_ridge:.2%}")

            self.log_to_console("Modelos de regressão treinados com sucesso!")

        except Exception as e:
            self.log_to_console(f"Erro ao treinar modelos de regressão: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao treinar modelos de regressão: {str(e)}")

    def prepare_classification(self):
        """Prepara dados para classificação"""
        try:
            self.log_to_console("\n=== PREPARANDO DADOS PARA CLASSIFICAÇÃO ===")

            # 1. Mapeamento de classes
            mapa_safra = {'baixa': 0, 'media': 1, 'alta': 2}
            self.df['safra_num'] = self.df['safra'].map(mapa_safra)

            # 2. Seleção de features
            self.X_class = self.df[['chuva_flor', 'chuva_colheita', 'chuva_total',
                                  'anomalia_flor', 'temp_flor', 'umid_flor', 'chuva_relativa']]
            self.y_class = self.df['safra_num'].fillna(1)

            # 3. Padronização
            self.scaler_class = StandardScaler()
            self.X_class_scaled = self.scaler_class.fit_transform(self.X_class)

            # 4. PCA
            self.pca_class = PCA(n_components=2)
            self.X_class_pca = self.pca_class.fit_transform(self.X_class_scaled)

            # 5. Divisão treino-teste
            self.X_train_class, self.X_test_class, self.y_train_class, self.y_test_class = train_test_split(
                self.X_class_scaled, self.y_class, test_size=0.3, random_state=42, stratify=self.y_class
            )
            
            # 6. PCA para treino e teste
            self.X_train_pca = self.pca_class.transform(self.X_train_class)
            self.X_test_pca = self.pca_class.transform(self.X_test_class)

            # 7. Binarização das classes para ROC
            classes = [0, 1, 2]
            self.y_test_bin = label_binarize(self.y_test_class, classes=classes)
            self.y_train_bin = label_binarize(self.y_train_class, classes=classes)
            n_classes = len(classes)

            self.log_to_console("Dados preparados para classificação!")
            self.log_to_console(f"Classes: {mapa_safra}")
            self.log_to_console(f"Distribuição das classes:\n{self.y_class.value_counts().sort_index()}")

            self.train_classification_btn.configure(state="normal")

        except Exception as e:
            self.log_to_console(f"Erro ao preparar classificação: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao preparar classificação: {str(e)}")

    def train_classification(self):
        """Treina modelos de classificação"""
        try:
            self.log_to_console("\n=== TREINANDO MODELOS DE CLASSIFICAÇÃO ===")

            # 1. Modelo Clássico
            self.modelo_classico = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            self.modelo_classico.fit(self.X_train_class, self.y_train_class)
            self.y_pred_classico = self.modelo_classico.predict(self.X_test_class)
            self.y_prob_classico = self.modelo_classico.predict_proba(self.X_test_class)

            # 2. Modelo PCA
            self.modelo_pca_class = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            self.modelo_pca_class.fit(self.X_train_pca, self.y_train_class)
            self.y_pred_pca = self.modelo_pca_class.predict(self.X_test_pca)
            self.y_prob_pca = self.modelo_pca_class.predict_proba(self.X_test_pca)

            # 3. Métricas
            acc_classico = accuracy_score(self.y_test_class, self.y_pred_classico)
            acc_pca = accuracy_score(self.y_test_class, self.y_pred_pca)
            
            self.log_to_console(f"Acurácia do modelo clássico: {acc_classico:.2%}")
            self.log_to_console(f"Acurácia do modelo PCA: {acc_pca:.2%}")
            
            self.log_to_console("\nRelatório de Classificação - Modelo Clássico:")
            self.log_to_console(classification_report(self.y_test_class, self.y_pred_classico))
            
            self.log_to_console("\nRelatório de Classificação - Modelo PCA:")
            self.log_to_console(classification_report(self.y_test_class, self.y_pred_pca))

        except Exception as e:
            self.log_to_console(f"Erro ao treinar modelos de classificação: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao treinar modelos de classificação: {str(e)}")

    def run_qlearning(self):
        """Executa o algoritmo de aprendizado por reforço Q-learning"""
        try:
            self.log_to_console("\n=== EXECUTANDO APRENDIZADO POR REFORÇO (Q-LEARNING) ===")

            alpha = 0.9
            gamma = 0.9
            epsilon = float(self.rl_epsilon.get())

            modo = self.rl_mode.get()
            num_episodios = int(self.rl_episodes.get())

            q_table = {
                'muito_seco': {'muita_agua': 0.0, 'regar': 0.0, 'pouca_agua': 0.0, 'nao_regar': 0.0},
                'seco': {'muita_agua': 0.0, 'regar': 0.0, 'pouca_agua': 0.0, 'nao_regar': 0.0},
                'ideal': {'muita_agua': 0.0, 'regar': 0.0, 'pouca_agua': 0.0, 'nao_regar': 0.0},
                'encharcado': {'muita_agua': 0.0, 'regar': 0.0, 'pouca_agua': 0.0, 'nao_regar': 0.0},
            }

            def transicao(estado, acao):
                if estado == 'muito_seco':
                    if acao == 'pouca_agua': return 'muito_seco'
                    elif acao == 'regar': return 'seco'
                    elif acao == 'muita_agua': return 'ideal'
                    else: return 'muito_seco'
                elif estado == 'seco':
                    if acao == 'regar': return 'ideal'
                    elif acao == 'pouca_agua': return 'seco'
                    elif acao == 'muita_agua': return 'encharcado'
                    else: return 'seco'
                elif estado == 'ideal':
                    if acao == 'regar': return 'encharcado'
                    elif acao == 'pouca_agua': return 'ideal'
                    elif acao == 'muita_agua': return 'encharcado'
                    else: return 'seco'
                elif estado == 'encharcado':
                    if acao == 'regar': return 'encharcado'
                    elif acao == 'pouca_agua': return 'ideal'
                    elif acao == 'muita_agua': return 'encharcado'
                    else: return 'ideal'

            def recompensa(estado, acao, modo='padrão'):
                padrao = {
                    'muito_seco': {'regar': 3, 'pouca_agua': 1, 'nao_regar': -1, 'muita_agua': 5},
                    'seco': {'regar': 5, 'pouca_agua': 2, 'nao_regar': -1, 'muita_agua': -3},
                    'ideal': {'nao_regar': 5, 'pouca_agua': 2, 'regar': -3, 'muita_agua': -5},
                    'encharcado': {'nao_regar': 2, 'pouca_agua': -1, 'regar': -3, 'muita_agua': -5},
                }
                extremo = {
                    'muito_seco': {'regar': 6, 'pouca_agua': 0, 'nao_regar': -3, 'muita_agua': 8},
                    'seco': {'regar': 8, 'pouca_agua': 5, 'nao_regar': -4, 'muita_agua': -6},
                    'ideal': {'nao_regar': -2, 'pouca_agua': 4, 'regar': 7, 'muita_agua': -4},
                    'encharcado': {'nao_regar': -4, 'pouca_agua': -2, 'regar': 3, 'muita_agua': -6},
                }
                tabelas = {'padrão': padrao, 'extremo': extremo}
                if modo not in tabelas:
                    raise ValueError("Modo inválido. Use 'padrão' ou 'extremo'.")
                return tabelas[modo][estado].get(acao, -10)

            historico = []

            for episodio in range(1, num_episodios):
                estado = random.choice(['muito_seco', 'seco', 'ideal', 'encharcado'])
                for passo in range(1):
                    if random.random() < epsilon:
                        acao = random.choice(['muita_agua', 'regar', 'pouca_agua', 'nao_regar'])
                    else:
                        acao = max(q_table[estado], key=q_table[estado].get)
                    prox_estado = transicao(estado, acao)
                    r = recompensa(estado, acao, modo)

                    max_q_prox = max(q_table[prox_estado].values())
                    q_atual = q_table[estado][acao]
                    q_novo = q_atual + alpha * (r + gamma * max_q_prox - q_atual)
                    q_table[estado][acao] = q_novo

                    historico.append({
                        'Episódio': episodio,
                        'Estado': estado,
                        'Ação': acao,
                        'Recompensa': r,
                        'Próximo estado': prox_estado,
                        'Q(s,a)': round(q_novo, 2)
                    })
                    estado = prox_estado

            q_df = pd.DataFrame(q_table).T
            self.log_to_console("\nTabela final de Q-values:")
            self.log_to_console(str(q_df))

            historico_df = pd.DataFrame(historico)
            self.log_to_console("\nÚltimas 10 decisões do histórico:")
            self.log_to_console(historico_df.tail(10).to_string(index=False))

        except Exception as e:
            self.log_to_console(f"Erro no Q-learning: {str(e)}")
            messagebox.showerror("Erro", f"Erro no Q-learning: {str(e)}")

    def run_complete_pipeline(self):
        """Executa todo o pipeline completo"""
        try:
            # Limpa gráficos anteriores
            self.clear_plots()
            self.log_to_console("\n=== INICIANDO PIPELINE COMPLETO ===")
            
            # Executa todas as etapas
            self.load_data()
            self.create_additional_vars()
            
            # Gera e armazena todos os gráficos EDA
            self.log_to_console("\nGerando visualizações exploratórias...")
            self.boxplot_enso()
            self.scatterplot_temp()
            self.histogram_vars()
            self.heatmap_correlation()
            self.pairplot_vars()
            
            # Continua com o resto do pipeline
            self.log_to_console("\nPreparando modelos de regressão...")
            self.prepare_regression()  # Já inclui o PCA e Scree Plot
            self.train_regression()
            
            self.log_to_console("\nPreparando modelos de classificação...")
            self.prepare_classification()
            self.train_classification()
            
            self.log_to_console("\nExecutando Q-learning...")
            self.run_qlearning()
            
            self.log_to_console("\n=== PIPELINE COMPLETO FINALIZADO ===")
            self.results_text.insert("end", "Pipeline completo executado com sucesso!\n")
            self.results_text.see("end")
            
            # Exibe o primeiro gráfico se houver algum
            if self.plots:
                self.current_plot_index = 0
                self.display_current_plot()
            else:
                self.log_to_console("Nenhum gráfico foi gerado durante o pipeline.")
                
        except Exception as e:
            self.log_to_console(f"Erro no pipeline completo: {str(e)}")
            messagebox.showerror("Erro", f"Erro no pipeline completo: {str(e)}")

    def scree_plot(self):
        """Cria Scree Plot para análise PCA"""
        try:
            if not hasattr(self, 'X_padronizado'):
                self.log_to_console("Execute 'Preparar Regressão' primeiro para gerar o Scree Plot.")
                return None

            if not hasattr(self, 'pca_full'):
                self.log_to_console("PCA não foi executado. Execute 'Preparar Regressão' primeiro.")
                return None

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                range(1, len(self.pca_full.explained_variance_ratio_) + 1),
                self.pca_full.explained_variance_ratio_,
                marker='o'
            )
            ax.set_title('Scree Plot - Variância Explicada por Componente')
            ax.set_xlabel('Componente Principal')
            ax.set_ylabel('Proporção da Variância')
            ax.grid(True)
            plt.tight_layout()
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar Scree Plot: {str(e)}")
            return None

    def plot_rmse_lambda(self):
        """Plota comparação do RMSE em função de lambda"""
        try:
            # Dados simulados para plotagem
            lambdas = [0.1, 1, 10, 100, 1000, 10000, 30000, 100000, 300000, 1000000]
            rmse_sem_pca = [71.19, 68.30, 54.97, 34.91, 26.38, 25.61, 25.56, 25.55, 25.54, 25.54]
            rmse_com_pca = [43.25, 42.98, 40.61, 31.20, 25.97, 25.57, 25.55, 25.54, 25.54, 25.54]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(lambdas, rmse_sem_pca, marker='o', label='Sem PCA')
            ax.plot(lambdas, rmse_com_pca, marker='s', label='Com PCA')
            ax.set_xscale('log')
            ax.set_xlabel("λ (log scale)")
            ax.set_ylabel("RMSE")
            ax.set_title("Comparação do RMSE em função de λ (Ridge)")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar gráfico RMSE vs Lambda: {str(e)}")
            return None

    def plot_cost_function_1d(self):
        """Plota função de custo 1D para uma variável"""
        try:
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                self.log_to_console("Execute 'Carregar Dados' primeiro para gerar a função de custo.")
                return None

            x_var = 'temp_flor'
            if x_var not in self.X.columns:
                self.log_to_console(f"Variável {x_var} não encontrada nos dados.")
                return None

            x = self.X[x_var].values
            y = self.y.values
            m = len(y)
            x_centralizado = x - x.mean()
            theta1_vals = np.linspace(-200, 200, 200)
            custos = [(1 / (2 * m)) * np.sum((theta1 * x_centralizado - y) ** 2) for theta1 in theta1_vals]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(theta1_vals, custos)
            ax.set_xlabel("θ₁")
            ax.set_ylabel("J(θ₁)")
            ax.set_title(f"Função de Custo - {x_var} (x centralizado)")
            ax.grid(True)
            plt.tight_layout()
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar função de custo 1D: {str(e)}")
            return None

    def plot_cost_function_2d(self):
        """Plota função de custo 2D para duas variáveis"""
        try:
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                self.log_to_console("Execute 'Carregar Dados' primeiro para gerar a função de custo.")
                return None

            x_vars = ['temp_flor', 'chuva_flor']
            for var in x_vars:
                if var not in self.X.columns:
                    self.log_to_console(f"Variável {var} não encontrada nos dados.")
                    return None

            x1 = self.X[x_vars[0]].values
            x2 = self.X[x_vars[1]].values
            y = self.y.values
            m = len(y)
            X_mat = np.vstack([np.ones(m), x1, x2]).T

            theta1_vals = np.linspace(-200, 200, 100)
            theta2_vals = np.linspace(-200, 200, 100)
            J_vals = np.zeros((100, 100))

            for i in range(100):
                for j in range(100):
                    theta = np.array([0, theta1_vals[i], theta2_vals[j]])
                    h = X_mat @ theta
                    J_vals[j, i] = (1 / (2 * m)) * np.sum((h - y) ** 2)

            T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(T1, T2, J_vals, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel(f"θ₁ ({x_vars[0]})")
            ax.set_ylabel(f"θ₂ ({x_vars[1]})")
            ax.set_zlabel("J(θ)")
            ax.set_title(f"Superfície da Função de Custo — {x_vars[0]} e {x_vars[1]}")
            fig.subplots_adjust(right=0.85)
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar função de custo 2D: {str(e)}")
            return None

    def plot_cost_function_2d_pca(self):
        """Plota função de custo 2D para componentes PCA"""
        try:
            if not hasattr(self, 'X_pca'):
                self.log_to_console("Execute 'Preparar Regressão' primeiro para gerar a função de custo PCA.")
                return None

            if self.X_pca.shape[1] < 2:
                self.log_to_console("PCA precisa ter pelo menos 2 componentes. Execute 'Preparar Regressão' primeiro.")
                return None

            pc1 = self.X_pca[:, 0]
            pc2 = self.X_pca[:, 1]
            y = self.y.values
            m = len(y)
            X_mat = np.vstack([np.ones(m), pc1, pc2]).T

            theta1_vals = np.linspace(-200, 200, 100)
            theta2_vals = np.linspace(-200, 200, 100)
            J_vals = np.zeros((100, 100))

            for i in range(100):
                for j in range(100):
                    theta = np.array([0, theta1_vals[i], theta2_vals[j]])
                    h = X_mat @ theta
                    J_vals[j, i] = (1 / (2 * m)) * np.sum((h - y) ** 2)

            T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(T1, T2, J_vals, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel("θ₁ (PC1)")
            ax.set_ylabel("θ₂ (PC2)")
            ax.set_zlabel("J(θ)")
            ax.set_title("Superfície da Função de Custo — Componentes Principais (PCA)")
            fig.subplots_adjust(right=0.85)
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar função de custo 2D PCA: {str(e)}")
            return None

    def plot_residuals(self):
        """Plota gráficos de resíduos para todos os modelos"""
        try:
            if not hasattr(self, 'y'):
                self.log_to_console("Execute 'Carregar Dados' primeiro.")
                return None

            if not all(hasattr(self, attr) for attr in ['modelo_linear', 'modelo_ridge', 'modelo_pca', 'modelo_pca_ridge']):
                self.log_to_console("Execute 'Treinar Regressão' primeiro para gerar os gráficos de resíduos.")
                return None

            # Previsões
            y_pred_linear = self.modelo_linear.predict(self.X_scaled)
            y_pred_ridge = self.modelo_ridge.predict(self.X_scaled)
            y_pred_pca = self.modelo_pca.predict(self.X_pca)
            y_pred_pca_ridge = self.modelo_pca_ridge.predict(self.X_pca)

            # Cria figura com subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            # Função auxiliar para plotar resíduos
            def plot_residuos_ax(y_true, y_pred, titulo, ax):
                residuos = y_true - y_pred
                ax.scatter(y_pred, residuos, color='royalblue', alpha=0.7)
                ax.axhline(0, color='red', linestyle='--')
                ax.set_xlabel("Previsão")
                ax.set_ylabel("Resíduo")
                ax.set_title(f"Resíduos — {titulo}")
                ax.grid(True)

            # Plota resíduos para cada modelo
            plot_residuos_ax(self.y, y_pred_linear, "Regressão Linear", axes[0])
            plot_residuos_ax(self.y, y_pred_ridge, "Regressão Regularizada (λ = 1.000.000)", axes[1])
            plot_residuos_ax(self.y, y_pred_pca, "PCA + Regressão Linear", axes[2])
            plot_residuos_ax(self.y, y_pred_pca_ridge, "PCA + Regularizada (λ = 100.000)", axes[3])

            plt.tight_layout()
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar gráficos de resíduos: {str(e)}")
            return None

    def plot_pca_2d(self):
        """Visualiza os dados em 2D usando os componentes principais"""
        try:
            if not hasattr(self, 'df_PCA'):
                self.log_to_console("Execute 'Preparar Regressão' primeiro para gerar a visualização PCA 2D.")
                return None

            if 'PC1' not in self.df_PCA.columns or 'PC2' not in self.df_PCA.columns:
                self.log_to_console("Componentes PCA não encontrados. Execute 'Preparar Regressão' primeiro.")
                return None

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=self.df_PCA, x='PC1', y='PC2', hue='safra', s=80, alpha=0.8, ax=ax)
            ax.set_title('PCA - Componentes Principais coloridos por Safra')
            ax.set_xlabel('Componente Principal 1')
            ax.set_ylabel('Componente Principal 2')
            ax.legend(title='Safra')
            plt.tight_layout()
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar visualização PCA 2D: {str(e)}")
            return None

    def plot_decision_boundaries(self):
        """Plota fronteiras de decisão do modelo com PCA"""
        try:
            if not hasattr(self, 'X_class_pca'):
                self.log_to_console("Execute 'Preparar Classificação' primeiro.")
                return None

            if not hasattr(self, 'modelo_pca_class'):
                self.log_to_console("Execute 'Treinar Classificação' primeiro para gerar as fronteiras de decisão.")
                return None

            if self.X_class_pca.shape[1] < 2:
                self.log_to_console("PCA precisa ter pelo menos 2 componentes. Execute 'Preparar Classificação' primeiro.")
                return None

            h = 0.02  # Passo da malha
            
            # Geração da malha de pontos
            x_min, x_max = self.X_class_pca[:, 0].min() - 1, self.X_class_pca[:, 0].max() + 1
            y_min, y_max = self.X_class_pca[:, 1].min() - 1, self.X_class_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Predição sobre a malha
            Z = self.modelo_pca_class.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Paleta de cores
            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
            cmap_bold = ['red', 'green', 'blue']
            
            # Gráfico
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
            scatter = ax.scatter(self.X_class_pca[:, 0], self.X_class_pca[:, 1], 
                               c=self.y_class, cmap=ListedColormap(cmap_bold), 
                               edgecolor='k', s=60)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Fronteiras de Decisão — PCA + Regressão Logística")
            ax.legend(handles=scatter.legend_elements()[0], labels=['Baixa', 'Média', 'Alta'])
            ax.grid(True)
            plt.tight_layout()
            self.show_plot(fig)
            return fig

        except Exception as e:
            self.log_to_console(f"Erro ao criar fronteiras de decisão: {str(e)}")
            return None

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    show_login_screen()
