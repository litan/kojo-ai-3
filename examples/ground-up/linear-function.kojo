// #include /plot.kojo

cleari()
clearOutput()

val seed = 42
initRandomGenerator(seed)

val m = 3
val c = 1

def f(x: Double) = m * x + c

repeatFor(0 to 20) { x =>
    val y = f(x)
    println(s"x=$x, f(x)=$y")
}

val xData = Array.tabulate(11)(n => n.toDouble)
val yData = xData.map(x => f(x) + randomDouble(-3, 3))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)
