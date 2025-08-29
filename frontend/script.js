const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const fileInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const heatmap = document.getElementById("heatmap");
const resultDiv = document.getElementById("result");
const spinner = document.getElementById("spinner");
const downloads = document.getElementById("downloads");
const downloadHeatmap = document.getElementById("downloadHeatmap");
const downloadResult = document.getElementById("downloadResult");

predictBtn.addEventListener("click", function () {
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image.");
    return;
  }

  // Reset state
  resultDiv.innerHTML = "";
  heatmap.style.display = "none";
  preview.style.display = "none";
  heatmap.classList.remove("show");
  preview.classList.remove("show");
  spinner.style.display = "block";

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      spinner.style.display = "none";

      resultDiv.innerHTML = `
        Prediction: <strong style="color: ${data.prediction === 'Pneumonia' ? 'red' : 'green'}">
          ${data.prediction}</strong><br>
        Confidence: <strong>${data.confidence}</strong>
      `;

      heatmap.src = data.heatmap_url;
      heatmap.style.display = "block";

      // Show download buttons
      downloads.style.display = "block";
      downloadHeatmap.href = data.heatmap_url;

      const resultText = `Prediction: ${data.prediction}\nConfidence: ${data.confidence}`;
      const blob = new Blob([resultText], { type: "text/plain" });
      const blobURL = URL.createObjectURL(blob);
      downloadResult.href = blobURL;

      setTimeout(() => {
        preview.classList.add("show");
        heatmap.classList.add("show");
      }, 100);
    })
    .catch(err => {
      spinner.style.display = "none";
      console.error("Error:", err);
      resultDiv.innerText = "Something went wrong.";
    });
});

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  preview.src = "#";
  heatmap.src = "#";
  preview.style.display = "none";
  heatmap.style.display = "none";
  resultDiv.innerHTML = "";
  preview.classList.remove("show");
  heatmap.classList.remove("show");
  downloads.style.display = "none";
});
