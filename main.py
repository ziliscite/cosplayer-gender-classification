import gui

if __name__ == "__main__":
  demo = gui.create_interface()
  demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True 
  )
