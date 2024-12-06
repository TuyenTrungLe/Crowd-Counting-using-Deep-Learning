let imageElement = null;
let imageGallery = [];

function uploadImage(event) {
  const file = event.target.files[0];
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

        addImageToGallery(imageElement.src);
      };
    };
    reader.readAsDataURL(file);
  } else {
    alert("No image selected.");
  }
}

function addImageToGallery(imageSrc) {
  imageGallery.push(imageSrc);

  const imageContainer = document.createElement("div");
  imageContainer.classList.add("image-item");
  const img = document.createElement("img");
  img.src = imageSrc;
  img.style.transform = "scale(0.75)";
  img.alt = "Uploaded Image";

  const deleteBtn = document.createElement("button");
  deleteBtn.classList.add("delete-btn");
  deleteBtn.innerText = "X";
  deleteBtn.onclick = function () {
    deleteImage(imageSrc, imageContainer);
  };

  imageContainer.appendChild(img);
  imageContainer.appendChild(deleteBtn);

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
