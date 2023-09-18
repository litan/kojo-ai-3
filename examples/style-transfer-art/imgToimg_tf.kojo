// #include /nst-tf.kojo

cleari()
clearOutput()

val baseDir = kojoCtx.baseDir
val modelDir = s"$baseDir/nst_model_gh/"
val styleImage = s"$baseDir/style/woman_with_hat_matisse_cropped.jpg"
val contentImage = s"$baseDir/content/cornell_cropped.jpg"

def checkExists(filename: String, desc: String) {
    import java.io.File
    require(new File(filename).exists, s"$desc does not exist")
}

checkExists(modelDir, "Model directory")
checkExists(styleImage, "Style image")
checkExists(contentImage, "Content image")

val alpha = 0.7f
timeit("NST TensorFlow") {
    val style = new NeuralStyleFilter(modelDir, styleImage, alpha)
    val content = Picture.image(contentImage)
    val pic = content.withEffect(style)
    drawCentered(pic)
}
