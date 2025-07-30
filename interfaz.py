import tkinter as tk

def dummy_func():
    print("Botón presionado")

ventana = tk.Tk()
ventana.title("Prueba interfaz")

tk.Label(ventana, text="Pega la URL de la imagen:").pack(pady=10)
entrada_url = tk.Entry(ventana, width=50)
entrada_url.pack(pady=10)

tk.Button(ventana, text="Probar botón", command=dummy_func).pack(pady=10)

ventana.mainloop()
