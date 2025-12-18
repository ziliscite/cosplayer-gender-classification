from inference import classify_image
import gradio as gr

def create_interface():
  # Custom CSS for styling
  custom_css = """
  .gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
  }
  .preprocessing-steps {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    margin: 10px 0;
  }
  .result-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
  }
  h1 {
    text-align: center;
    color: #764ba2;
    font-weight: 700;
  }
  .step-title {
    font-weight: 600;
    color: #667eea;
    margin-bottom: 8px;
  }
  """
    
  with gr.Blocks(css=custom_css, title="Image Classifier") as demo:
    gr.Markdown(
      """
      # Cosplayer Gender Classification Demo
      
      Upload an image to see the preprocessing steps and classification results.
      This demo shows preprocessing stages before the final verdict.
      """
    )
    
    with gr.Row():
      with gr.Column(scale=1):
        gr.Markdown("### Input")
        input_image = gr.Image(
          type="numpy",
          label="Upload Image",
          height=500
        )
        classify_btn = gr.Button(
          "🔍 Classify Image",
          variant="primary",
          size="lg"
        )
          
        gr.Markdown(
          """
          ---
          ### Instructions:
          1. Upload an image using the box above
          2. Click the "Classify Image" button
          3. View preprocessing steps and results
          """
        )
      
      with gr.Column(scale=2):
        gr.Markdown("### Preprocessing Steps")
        
        with gr.Column(scale=1):
          step1_output = gr.Image(
            label="Step 1: Detect Faces",
            height=400
          )

        with gr.Column(scale=2):
          with gr.Row():
            step2_output = gr.Image(
              label="Step 2: Crop Faces",
              height=200
            )
            step3_output = gr.Image(
              label="Step 3: Resize to 128x128px",
              height=200
            )
            step4_output = gr.Image(
              label="Step 4: Gray Scale Conversion",
              height=200
            )
            step5_output = gr.Image(
              label="Step 5: Histogram Equalization",
              height=200
            )
        
        gr.Markdown("### Classification Results")
        result_output = gr.HTML(
          value="""
          <div class="result-box">
            <h2 style="color: white; margin: 0; font-size: 1.2em;">Awaiting classification...</h2>
          </div>
          """,
          label="Results"
        )
    
    # wrapper function to format the label result
    def classify_and_format(image):
      try:
        step1, step2, step3, step4, step5, result = classify_image(image)
        if result:
          result_text = result.strip().upper()
        
          if "FEMALE" in result_text:
            icon = "👩"
            gradient = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
            label = "FEMALE"
          else:
            icon = "👨"
            gradient = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
            label = "MALE"
          
          formatted_result = f"""
            <div style="background: {gradient}; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
              <div style="font-size: 5em; margin-bottom: 10px;">{icon}</div>
              <h1 style="color: white; margin: 0; font-size: 3em; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{label}</h1>
            </div>
            """
        else:
            formatted_result = """
            <div class="result-box">
              <h2 style="color: white; margin: 0;">No result available</h2>
            </div>
            """  
        return step1, step2, step3, step4, step5, formatted_result
      
      except Exception as e:
        return None, None, None, None, None, """
          <div class="result-box
            <h2 style="color: white; margin: 0;">No face detected in the image.</h2>
          </div>
          """
    
    # Set up the event handler
    classify_btn.click(
      fn=classify_and_format,
      inputs=[input_image],
      outputs=[step1_output, step2_output, step3_output, step4_output, step5_output, result_output]
    )
  
  return demo