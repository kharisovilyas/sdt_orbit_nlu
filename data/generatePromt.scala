import java.io.{File, PrintWriter}
import scala.util.Random
import java.time.LocalDate
import java.time.format.DateTimeFormatter

object GeneratePrompt {
  case class Prompt(prompt: String, filters: Map[String, String]) {
    def toJson: String = {
      val filtersJson = filters.map { case (k, v) => s""""$k": "$v"""" }.mkString(", ")
      s"""{"prompt": "$prompt", "filters": { $filtersJson }}"""
    }
  }

  val rnd = new Random()
  val formatter = DateTimeFormatter.ISO_DATE

  // Define data with proper case forms
  val coverages = Map(
    "Россия" -> Map("nominative" -> "Россия", "accusative" -> "Россию", "prepositional" -> "России"),
    "Арктика" -> Map("nominative" -> "Арктика", "accusative" -> "Арктику", "prepositional" -> "Арктике"),
    "Африка" -> Map("nominative" -> "Африка", "accusative" -> "Африку", "prepositional" -> "Африке"),
    "Китай" -> Map("nominative" -> "Китай", "accusative" -> "Китай", "prepositional" -> "Китае"),
    "Европа" -> Map("nominative" -> "Европа", "accusative" -> "Европу", "prepositional" -> "Европе"),
    "Южная Америка" -> Map("nominative" -> "Южная Америка", "accusative" -> "Южную Америку", "prepositional" -> "Южной Америке")
  )

  val altitudes = Seq("<600 км", "<1000 км", "500-800 км", "~800 км", "700-900 км")
  
  val orbitTypesMap = Map(
    "LEO" -> Map("nominative" -> "низкая околоземная орбита", "accusative" -> "низкую околоземную орбиту", "prepositional" -> "низкой околоземной орбите"),
    "MEO" -> Map("nominative" -> "средняя околоземная орбита", "accusative" -> "среднюю околоземную орбиту", "prepositional" -> "средней околоземной орбите"),
    "GEO" -> Map("nominative" -> "геостационарная орбита", "accusative" -> "геостационарную орбиту", "prepositional" -> "геостационарной орбите"),
    "SSO" -> Map("nominative" -> "солнечно-синхронная орбита", "accusative" -> "солнечно-синхронную орбиту", "prepositional" -> "солнечно-синхронной орбите"),
    "Molniya" -> Map("nominative" -> "орбита Молния", "accusative" -> "орбиту Молния", "prepositional" -> "орбите Молния"),
    "HEO" -> Map("nominative" -> "высокая эллиптическая орбита", "accusative" -> "высокую эллиптическую орбиту", "prepositional" -> "высокой эллиптической орбите")
  )

  val statuses = Map(
    "активен" -> Map("nominative" -> "активен", "accusative" -> "активный"),
    "неактивен" -> Map("nominative" -> "неактивен", "accusative" -> "неактивный")
  )

  val formFactors = Seq("1U", "2U", "3U", "6U", "12U", "24U", "36U", "48U")
  val massRange = 10 to 200

  // Updated templates with explicit case requirements
  val templates = Seq(
    ("Подбери {satelliteText} с форм-фактором {formFactor}, которые видят {coverage:accusative} и весят {mass}.", Map("coverage" -> "accusative")),
    ("Группировка из {satelliteText} на круговых орбитах {altitude}, часто пересекающих {coverage:accusative}.", Map("coverage" -> "accusative")),
    ("Нужен {satelliteText} с форм-фактором {formFactor}, типом орбиты {orbitType:accusative}, статус {status:accusative}.", Map("orbitType" -> "accusative", "status" -> "accusative")),
    ("Подбери {scale} {satelliteText} на орбите {orbitType:prepositional}, рассчитанные по TLE от {tleDate}.", Map("orbitType" -> "prepositional")),
    ("Составь список {satelliteText}, которые работают на высоте {altitude} и наблюдают {coverage:accusative}.", Map("coverage" -> "accusative")),
    ("Найди {satelliteText} с массой {mass}, расположенные на орбите {orbitType:prepositional} над {coverage:prepositional}.", Map("orbitType" -> "prepositional", "coverage" -> "prepositional"))
  )

  // Helper functions
  def satelliteWordForm(number: Int): String = number match {
    case 1 => "спутник"
    case 2 | 3 | 4 => "спутника"
    case _ => "спутников"
  }

  def scaleWordForm(scale: String, number: Int): String = number match {
    case 1 => scale
    case _ => if (scale == "малый") "малые" else "большие"
  }

  def randomDate(): String = LocalDate.now().minusDays(rnd.nextInt(365 * 3)).format(formatter)

  def determineScale(mass: Int, formFactor: String): String = {
    val massScale = if (mass <= 100) "малый" else "большой"
    val formScale = if (formFactor.replace("U", "").toInt <= 24) "малый" else "большой"
    if (massScale == formScale) massScale else "большой"
  }

  def randomOrbitType(caseType: String): String = {
    val key = orbitTypesMap.keys.toSeq(rnd.nextInt(orbitTypesMap.size))
    orbitTypesMap(key)(caseType)
  }

  def randomCoverage(caseType: String): String = {
    val key = coverages.keys.toSeq(rnd.nextInt(coverages.size))
    coverages(key)(caseType)
  }

  def randomStatus(caseType: String): String = {
    val key = statuses.keys.toSeq(rnd.nextInt(statuses.size))
    statuses(key)(caseType)
  }

  def randomSatelliteNumber(): Int = {
    val p = rnd.nextDouble()
    if (p < 0.5) 1
    else if (p < 0.9) 2 + rnd.nextInt(9)
    else 11 + rnd.nextInt(9990)
  }

  def numberToWords(number: Int): String = {
    if (number == 0) return "ноль"
    val units = Array("", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
    val teens = Array("десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
      "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать")
    val tens = Array("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
      "семьдесят", "восемьдесят", "девяносто")
    val hundreds = Array("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
      "семьсот", "восемьсот", "девятьсот")
    val sb = new StringBuilder
    var n = number

    if (n >= 1000) {
      val th = n / 1000
      sb.append(if (th == 1) "одна тысяча"
      else if (th == 2) "две тысячи"
      else numberToWords(th) + " тысяч")
      sb.append(" ")
      n = n % 1000
    }

    if (n >= 100) {
      sb.append(hundreds(n / 100)).append(" ")
      n = n % 100
    }

    if (n >= 20) {
      sb.append(tens(n / 10)).append(" ")
      n = n % 10
    } else if (n >= 10) {
      sb.append(teens(n - 10)).append(" ")
      n = 0
    }

    if (n > 0) sb.append(units(n)).append(" ")
    sb.toString.trim
  }

  def satelliteText(number: Int): String = {
    val useWord = rnd.nextBoolean()
    val word = satelliteWordForm(number)
    if (useWord) s"${numberToWords(number)} $word" else s"$number $word"
  }

  def generatePrompt(): Prompt = {
    val (template, caseRequirements) = templates(rnd.nextInt(templates.length))
    val coverageKey = coverages.keys.toSeq(rnd.nextInt(coverages.size))
    val orbitTypeKey = orbitTypesMap.keys.toSeq(rnd.nextInt(orbitTypesMap.size))
    val statusKey = statuses.keys.toSeq(rnd.nextInt(statuses.size))
    val altitude = altitudes(rnd.nextInt(altitudes.length))
    val formFactor = formFactors(rnd.nextInt(formFactors.length))
    val mass = massRange(rnd.nextInt(massRange.size))
    val scale = determineScale(mass, formFactor)
    val tleDate = randomDate()
    val numberOfSatellites = randomSatelliteNumber()
    val satelliteTextValue = satelliteText(numberOfSatellites)
    val scaleWord = scaleWordForm(scale, numberOfSatellites)

    // Replace placeholders with case-specific values
    val filledText = template.replace("{satelliteText}", satelliteTextValue)
      .replace("{altitude}", altitude)
      .replace("{formFactor}", formFactor)
      .replace("{mass}", s"$mass кг")
      .replace("{scale}", scaleWord)
      .replace("{tleDate}", tleDate)
      .replace("{coverage:accusative}", coverages(coverageKey)("accusative"))
      .replace("{coverage:prepositional}", coverages(coverageKey)("prepositional"))
      .replace("{orbitType:accusative}", orbitTypesMap(orbitTypeKey)("accusative"))
      .replace("{orbitType:prepositional}", orbitTypesMap(orbitTypeKey)("prepositional"))
      .replace("{status:accusative}", statuses(statusKey)("accusative"))

    val filters = Map(
      "coverage" -> coverageKey,
      "altitude" -> altitude,
      "orbitType" -> orbitTypeKey,
      "status" -> statusKey,
      "formFactor" -> formFactor,
      "mass" -> s"$mass кг",
      "scale" -> scaleWord,
      "tleDate" -> tleDate,
      "numberOfSatellites" -> numberOfSatellites.toString
    )

    Prompt(filledText, filters)
  }

  def main(args: Array[String]): Unit = {
    val count = if (args.nonEmpty) args(0).toInt else 500
    val seen = scala.collection.mutable.Set[String]()
    val prompts = scala.collection.mutable.ArrayBuffer[Prompt]()

    while (prompts.size < count) {
      val p = generatePrompt()
      val key = p.toJson
      if (!seen.contains(key)) {
        seen += key
        prompts += p
      }
    }

    val pw = new PrintWriter(new File("prompts.jsonl"), "UTF-8")
    try prompts.foreach(p => pw.println(p.toJson))
    finally pw.close()

    println(s"Сгенерировано ${prompts.size} уникальных промптов → prompts.jsonl")
  }
}