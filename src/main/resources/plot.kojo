import net.kogics.kojo.plot._

import org.knowm.xchart.internal.chartpart.Chart
import org.knowm.xchart.internal.series.Series
import org.knowm.xchart.style.Styler
import org.knowm.xchart.style.XYStyler

var currChartPic: Option[Picture] = None

def chartPic[A <: Styler, B <: Series](chart: Chart[A, B]): Picture = {
    import org.knowm.xchart.BitmapEncoder
    val img = BitmapEncoder.getBufferedImage(chart)
    Picture.image(img)
}

def centeredChartPic[A <: Styler, B <: Series](chart: Chart[A, B]): Picture = {
    val cb = canvasBounds
    import org.knowm.xchart.BitmapEncoder
    val img = BitmapEncoder.getBufferedImage(chart)
    Picture.image(img)
        .withTranslation(
            cb.x + (cb.width - img.getWidth) / 2,
            cb.y + (cb.height - img.getHeight) / 2
        )
}

def drawChart[A <: Styler, B <: Series](chart: Chart[A, B]) {
    val newPic = centeredChartPic(chart)
    draw(newPic)
    currChartPic.foreach { pic =>
        pic.erase()
    }
    currChartPic = Some(newPic)
}

def updateChart[A <: Styler, B <: Series](chart: Chart[A, B]) {
    drawChart(chart)
}

def setChartRange[A <: XYStyler, B <: Series](
    chart: Chart[A, B],
    xmin:  Double, xmax: Double,
    ymin: Double, ymax: Double) {
    chart.getStyler.setXAxisMin(xmin)
    chart.getStyler.setXAxisMax(xmax)
    chart.getStyler.setYAxisMin(ymin)
    chart.getStyler.setYAxisMax(ymax)
}

class LiveChart(
    title:  String,
    xtitle: String, ytitle: String,
    xmin: Double, xmax: Double,
    ymin: Double, ymax: Double) {
    import org.knowm.xchart.XYChart

    val xsbuf = ArrayBuffer.empty[Double]
    val ysbuf = ArrayBuffer.empty[Double]

    def update(x: Double, y: Double) {
        xsbuf.append(x)
        ysbuf.append(y)
        val xs = xsbuf.toArray
        val ys = ysbuf.toArray
        if (xs.length > 1) {
            val chart = lineChart(title, xtitle, ytitle, xs, ys)
            setChartRange(chart, xmin, xmax, ymin, ymax)
            chart.getStyler.setXAxisDecimalPattern("0")
            drawChart(chart)
        }
    }
}
