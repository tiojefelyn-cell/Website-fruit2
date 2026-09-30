// =====================================
// CANVAS TRAINING
// =====================================

const canvas =
    document.getElementById("canvas");

const ctx =
    canvas.getContext("2d");

const testCanvas =
    document.getElementById("testCanvas");

const testCtx =
    testCanvas.getContext("2d");


// =====================================
// SETUP CANVAS
// =====================================

function setupCanvas(ctx) {

    ctx.fillStyle = "black";

    ctx.fillRect(
        0,
        0,
        280,
        280
    );

    ctx.strokeStyle = "white";

    ctx.lineWidth = 20;

    ctx.lineCap = "round";

    ctx.lineJoin = "round";
}

setupCanvas(ctx);
setupCanvas(testCtx);


// =====================================
// DRAW FUNCTION
// =====================================

function enableDrawing(canvas, ctx) {

    let drawing = false;

    function position(event) {

        const rect =
            canvas.getBoundingClientRect();

        let clientX;
        let clientY;

        if (event.touches) {

            clientX =
                event.touches[0].clientX;

            clientY =
                event.touches[0].clientY;

        } else {

            clientX =
                event.clientX;

            clientY =
                event.clientY;
        }

        return {

            x:
                (clientX - rect.left)
                *
                (canvas.width / rect.width),

            y:
                (clientY - rect.top)
                *
                (canvas.height / rect.height)

        };
    }


    function start(event) {

        event.preventDefault();

        drawing = true;

        const pos =
            position(event);

        ctx.beginPath();

        ctx.moveTo(
            pos.x,
            pos.y
        );
    }


    function draw(event) {

        event.preventDefault();

        if (!drawing) return;

        const pos =
            position(event);

        ctx.lineTo(
            pos.x,
            pos.y
        );

        ctx.stroke();
    }


    function stop() {

        drawing = false;
    }


    canvas.addEventListener(
        "mousedown",
        start
    );

    canvas.addEventListener(
        "mousemove",
        draw
    );

    canvas.addEventListener(
        "mouseup",
        stop
    );

    canvas.addEventListener(
        "mouseleave",
        stop
    );


    canvas.addEventListener(
        "touchstart",
        start,
        { passive: false }
    );

    canvas.addEventListener(
        "touchmove",
        draw,
        { passive: false }
    );

    canvas.addEventListener(
        "touchend",
        stop
    );
}

enableDrawing(canvas, ctx);

enableDrawing(testCanvas, testCtx);


// =====================================
// DATASET
// =====================================

const trainingImages = [];

const trainingLabels = [];

let targetNumber = 0;

let model = null;


// =====================================
// TARGET NUMBER
// =====================================

function changeTarget() {

    targetNumber =
        Math.floor(
            Math.random() * 10
        );

    document.getElementById(
        "targetNumber"
    ).textContent =
        targetNumber;
}

changeTarget();


// =====================================
// PREPROCESS IMAGE
// =====================================

function processCanvas(canvas) {

    return tf.tidy(() => {

        let image =
            tf.browser.fromPixels(
                canvas,
                1
            );

        image =
            tf.image.resizeBilinear(
                image,
                [28, 28]
            );

        image =
            image.div(255);

        image =
            image.expandDims(0);

        return image;
    });
}


// =====================================
// SAVE TRAINING EXAMPLE
// =====================================

document.getElementById(
    "saveBtn"
).addEventListener(
    "click",
    function () {

        const image =
            processCanvas(canvas);

        trainingImages.push(
            image.squeeze()
        );

        trainingLabels.push(
            targetNumber
        );

        document.getElementById(
            "dataStatus"
        ).textContent =
            "Data: "
            +
            trainingImages.length
            +
            " contoh";


        setupCanvas(ctx);

        changeTarget();

    }
);


// =====================================
// CLEAR TRAINING CANVAS
// =====================================

document.getElementById(
    "clearBtn"
).addEventListener(
    "click",
    function () {

        setupCanvas(ctx);

    }
);


// =====================================
// TRAIN AI
// =====================================

document.getElementById(
    "trainBtn"
).addEventListener(
    "click",
    async function () {

        if (
            trainingImages.length < 20
        ) {

            document.getElementById(
                "trainStatus"
            ).textContent =
                "⚠️ Buat minimal 20 contoh dulu.";

            return;
        }


        const trainBtn =
            document.getElementById(
                "trainBtn"
            );

        trainBtn.disabled = true;


        document.getElementById(
            "trainStatus"
        ).textContent =
            "🧠 AI sedang training...";


        try {

            const xs =
                tf.stack(
                    trainingImages
                );

            const labels =
                tf.tensor1d(
                    trainingLabels,
                    "int32"
                );

            const ys =
                tf.oneHot(
                    labels,
                    10
                );


            model =
                tf.sequential();


            model.add(
                tf.layers.flatten({
                    inputShape: [28, 28]
                })
            );


            model.add(
                tf.layers.dense({
                    units: 128,
                    activation: "relu"
                })
            );


            model.add(
                tf.layers.dense({
                    units: 64,
                    activation: "relu"
                })
            );


            model.add(
                tf.layers.dense({
                    units: 10,
                    activation: "softmax"
                })
            );


            model.compile({

                optimizer:
                    tf.train.adam(0.001),

                loss:
                    "categoricalCrossentropy",

                metrics:
                    ["accuracy"]

            });


            await model.fit(
                xs,
                ys,
                {

                    epochs: 30,

                    batchSize:
                        Math.min(
                            16,
                            trainingImages.length
                        ),

                    shuffle: true,

                    callbacks: {

                        onEpochEnd:
                            function (
                                epoch,
                                logs
                            ) {

                                const accuracy =
                                    (
                                        logs.acc * 100
                                    ).toFixed(1);

                                document.getElementById(
                                    "trainStatus"
                                ).textContent =
                                    "🧠 Epoch "
                                    +
                                    (epoch + 1)
                                    +
                                    "/30 — Akurasi "
                                    +
                                    accuracy
                                    +
                                    "%";

                            }

                    }

                }
            );


            xs.dispose();

            labels.dispose();

            ys.dispose();


            document.getElementById(
                "trainStatus"
            ).textContent =
                "✅ Training selesai! AI siap digunakan.";

        } catch (error) {

            console.error(error);

            document.getElementById(
                "trainStatus"
            ).textContent =
                "❌ Training gagal.";

        }


        trainBtn.disabled = false;

    }
);


// =====================================
// DETECT
// =====================================

document.getElementById(
    "detectBtn"
).addEventListener(
    "click",
    async function () {

        if (!model) {

            document.getElementById(
                "detectStatus"
            ).textContent =
                "⚠️ Training AI dulu.";

            return;
        }


        document.getElementById(
            "detectStatus"
        ).textContent =
            "🔍 AI sedang membaca...";


        document.getElementById(
            "resultNumber"
        ).textContent =
            "...";


        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    1500
                )
        );


        const input =
            processCanvas(
                testCanvas
            );


        const prediction =
            model.predict(input);


        const probabilities =
            await prediction.data();


        let bestIndex = 0;

        let bestProbability =
            probabilities[0];


        for (
            let i = 1;
            i < 10;
            i++
        ) {

            if (
                probabilities[i]
                >
                bestProbability
            ) {

                bestProbability =
                    probabilities[i];

                bestIndex =
                    i;

            }
        }


        document.getElementById(
            "resultNumber"
        ).textContent =
            bestIndex;


        document.getElementById(
            "confidence"
        ).textContent =
            "Confidence: "
            +
            (
                bestProbability * 100
            ).toFixed(1)
            +
            "%";


        document.getElementById(
            "detectStatus"
        ).textContent =
            "✅ Angka berhasil dikenali.";


        input.dispose();

        prediction.dispose();

    }
);