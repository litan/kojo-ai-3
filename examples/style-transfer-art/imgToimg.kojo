// #include /nst.kojo

cleari()
clearOutput()
setBackground(black)

val baseDir = kojoCtx.baseDir
val modelDir = s"$baseDir/nst_model_pt/"
val styleImage = s"$baseDir/style/ashville_cropped.jpg"
val contentImage = s"$baseDir/content/cornell_cropped.jpg"

def checkExists(filename: String, desc: String) {
    import java.io.File
    require(new File(filename).exists, s"$desc does not exist")
}

checkExists(modelDir, "Model directory")
checkExists(styleImage, "Style image")
checkExists(contentImage, "Content image")

val alpha = 0.9f
val style = new NeuralStyleFilter(modelDir, styleImage, alpha)
val content = Picture.image(contentImage)
val pic = content.withEffect(style)
val gap = 10
val out = picColCentered(
    pic,
    Picture.vgap(gap),
    picRowCentered(content, Picture.hgap(gap), Picture.image(styleImage))
).withScaling(0.75)
drawCentered(out)
