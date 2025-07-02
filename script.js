const canvas = document.getElementById("canvas")
const ctx = canvas.getContext("2d")
const clear = document.getElementById("clear")
const predict = document.getElementById("predict")
const prediction = document.getElementById("prediction")

let symbols = { 
    "+": "+",
    "-": "-",
    "!": "!",
    "alpha": "α",
    "beta": "β",
    "cos": "cos()",
    "Delta": "Δ",
    "div" : "÷",
    "exists": "∃",
    "forall": "∀",
    "gamma": "γ",
    "geq": "≥",
    "gt": ">",
    "i": "i",
    "in": "∈",
    "infty": "∞",
    "int" : "∫",
    "lambda": "λ",
    "ldots": "Idots",
    "leq": "≤",
    "lim": "lim()",
    "log": "log()",
    "lt": "<",
    "mu": "μ",
    "neq": "≠",
    "phi": "Φ",
    "pi": "π",
    "pm": "±",
    "rightarrow": "→",
    "sigma": "σ",
    "sin": "sin()",
    "sqrt": "√",
    "sum": "∑",
    "tan": "tan()",
    "theta": "θ",
    "times": "⋅",
    "=": "="
}

console.log(symbols["alpha"])

const url = "http://127.0.0.1:8000/"

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.drawImage(canvas, 0, 0);

clear.addEventListener('click', (e) => {
    clearCanvas()
})

predict.addEventListener('click', (e) => {
    sendImage()
})

let drawing = false
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (drawing) {
    ctx.lineWidth = 8;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
});

canvas.addEventListener('mouseleave', () => {
    drawing = false;
});

    
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(canvas, 0, 0);
}


function sendImage() {
    canvas.toBlob((blob) => {
        const data = new FormData()
        console.log(blob)
        data.append("file", blob)   
        fetch(url, {
            method: "POST",
            body: data,
        })
        .then((response) => {
            return response.json()
        })
        .then((data)=> {
            const response_array = JSON.parse(data.prediction)
            console.log(data.prediction)
            console.log(response_array.length)
            prediction.innerHTML = ""
            for (let i = 0; i < response_array.length; i++) {
                const button = document.createElement("button")
                button.textContent = symbols[response_array[i]]
                button.addEventListener("click", (e) => {
                navigator.clipboard.writeText(button.textContent);
                })
                prediction.appendChild(button)
            }   
        })
    })
}

