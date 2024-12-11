let imageElement = null;
let imageGallery = [];

function uploadImage(event) {
  const file = event.target.files[0];
  const estimateButton = document.getElementById("estimate-button");

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      imageElement = new Image();
      imageElement.src = e.target.result;
      imageElement.onload = function () {
        const imageBox = document.getElementById("image-box");
        imageBox.innerHTML = "";
        imageBox.style.display = "flex";
        imageBox.appendChild(imageElement);

        const imageAspectRatio = imageElement.width / imageElement.height;
        const containerWidth = imageBox.offsetWidth;
        const newHeight = containerWidth / imageAspectRatio;

        imageBox.style.height = newHeight + "px";
        estimateButton.disabled = false;
      };
    };
    reader.readAsDataURL(file);
  } else {
    alert("No image selected.");
  }
}

function estimate() {
  const fileInput = document.getElementById("file-upload");
  const file = fileInput.files[0];
  const result = document.getElementById("result");

  const formData = new FormData();
  formData.append("image", file);

  fetch("/estimate", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.prediction !== undefined) {
        addImageToGallery(imageElement.src, data.prediction);
        result.innerHTML =
          "Estimated number of people: " + data.prediction.toFixed(2);
      } else {
        alert("Error: " + data.error);
      }
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function addImageToGallery(imageSrc, prediction) {
  imageGallery.push(imageSrc);

  const imageContainer = document.createElement("div");
  imageContainer.classList.add("image-item");

  const img = document.createElement("img");
  img.src = imageSrc;
  img.style.transform = "scale(0.75)";
  img.alt = `Predicted count: ${prediction}`;

  const hoverText = document.createElement("div");
  hoverText.classList.add("hover-text");
  hoverText.innerText = `Estimated: ${prediction.toFixed(2)}`;

  const deleteBtn = document.createElement("button");
  deleteBtn.classList.add("delete-btn");
  deleteBtn.innerText = "X";
  deleteBtn.onclick = function () {
    deleteImage(imageSrc, imageContainer);
  };

  deleteBtn.addEventListener("mouseover", function () {
    hoverText.style.opacity = 0;
    hoverText.style.display = "none";
  });

  img.addEventListener("mouseover", function () {
    hoverText.style.opacity = 1;
    hoverText.style.display = "block";
  });

  img.addEventListener("mouseout", function () {
    hoverText.style.opacity = 0;
    hoverText.style.display = "none";
  });

  imageContainer.appendChild(img);
  imageContainer.appendChild(deleteBtn);
  imageContainer.appendChild(hoverText);

  const gallery = document.getElementById("image-gallery");
  gallery.appendChild(imageContainer);
}

function deleteImage(imageSrc, imageContainer) {
  imageGallery = imageGallery.filter((src) => src !== imageSrc);
  imageContainer.remove();
}

function showSection(section) {
  const sections = document.querySelectorAll("div[id]");
  sections.forEach((sec) => {
    sec.hidden = true;
  });

  document.getElementById(section).hidden = false;
}

showSection("home");
