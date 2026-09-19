#!/usr/bin/env python3
"""Generate two standalone academic research papers as PDFs."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    KeepTogether,
    ListFlowable,
    ListItem,
    HRFlowable,
)
from reportlab.lib.colors import HexColor, black


NAVY = HexColor("#1a365d")
RULE = HexColor("#2c5282")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Journal",
            fontName="Times-Italic",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperTitle",
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            textColor=black,
            spaceBefore=8,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Authors",
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Affiliation",
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            fontName="Times-Bold",
            fontSize=12,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor=NAVY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHead",
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            firstLineIndent=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AbstractBody",
            fontName="Times-Roman",
            fontSize=10,
            leading=13.5,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leftIndent=18,
            rightIndent=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AbstractLabel",
            fontName="Times-Bold",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Keywords",
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            leftIndent=18,
            rightIndent=18,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RefItem",
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12.5,
            leftIndent=18,
            firstLineIndent=-18,
            spaceAfter=5,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            fontName="Times-Italic",
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Footer",
            fontName="Times-Roman",
            fontSize=9,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Ack",
            fontName="Times-Italic",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    return styles


def header_footer(canvas, doc, running_title):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(0.85 * inch, letter[1] - 0.55 * inch, letter[0] - 0.85 * inch, letter[1] - 0.55 * inch)
    canvas.setFont("Times-Italic", 8.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(0.85 * inch, letter[1] - 0.48 * inch, running_title)
    canvas.setStrokeColor(RULE)
    canvas.line(0.85 * inch, 0.6 * inch, letter[0] - 0.85 * inch, 0.6 * inch)
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(black)
    canvas.drawCentredString(letter[0] / 2.0, 0.42 * inch, str(doc.page))
    canvas.restoreState()


def P(text, style):
    return Paragraph(text, style)


def build_doc(path, title, running_title, story):
    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        leftMargin=0.95 * inch,
        rightMargin=0.95 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.8 * inch,
        title=title,
        author="Independent Research Synthesis",
    )
    doc.build(
        story,
        onFirstPage=lambda c, d: header_footer(c, d, running_title),
        onLaterPages=lambda c, d: header_footer(c, d, running_title),
    )


def paper_one(styles, out_path):
    story = []
    story.append(P("OPEN RESEARCH MONOGRAPH &nbsp;&middot;&nbsp; ENVIRONMENTAL SCIENCE", styles["Journal"]))
    story.append(
        P(
            "Nature-Based Cooling and the Urban Heat Island Effect in Tropical Megacities: "
            "A Synthesis of Evidence, Trade-offs, and Implementation Pathways",
            styles["PaperTitle"],
        )
    )
    story.append(P("A. Rahman<sup>1</sup>, L. Chen<sup>2</sup>, and M. Okonkwo<sup>3</sup>", styles["Authors"]))
    story.append(
        P(
            "<sup>1</sup>Urban Climate Laboratory, Independent Synthesis Series &nbsp;&middot;&nbsp; "
            "<sup>2</sup>Department of Landscape Planning &nbsp;&middot;&nbsp; "
            "<sup>3</sup>Institute for Climate-Resilient Cities",
            styles["Affiliation"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=0.8, color=RULE, spaceAfter=8))
    story.append(P("Abstract", styles["AbstractLabel"]))
    story.append(
        P(
            "Tropical megacities concentrate population, impermeable surfaces, and waste heat in climates "
            "that already provide limited nocturnal relief. The urban heat island (UHI) therefore is not a "
            "marginal amenity issue but a public-health, energy, and equity problem. This paper synthesizes "
            "peer-reviewed evidence on nature-based cooling&mdash;street trees, urban forests, green roofs "
            "and walls, parks, blue-green corridors, and constructed wetlands&mdash;and evaluates how well "
            "these interventions reduce canopy, surface, and indoor temperatures under humid tropical "
            "conditions. We organize findings around four questions: (1) what magnitudes of cooling are "
            "consistently reported; (2) which biophysical mechanisms dominate in high humidity; (3) how "
            "benefits and burdens are distributed across neighborhoods; and (4) which governance and "
            "maintenance arrangements determine whether planted infrastructure survives the first decade. "
            "The synthesis indicates that shade from mature canopy is the most reliable daytime cooling "
            "pathway in the humid tropics, whereas evapotranspiration is often constrained by high ambient "
            "humidity and by water stress. Parks and water bodies produce local park-cool-island effects "
            "that decay rapidly with distance, implying that distributed canopy is more protective for "
            "pedestrians than a small number of large parks. Green roofs can reduce rooftop surface "
            "temperatures and building cooling loads, but their contribution to street-level thermal "
            "comfort is typically modest unless combined with street trees and reflective materials. "
            "We conclude with an implementation framework that treats cooling as a measurable municipal "
            "service, not as ornamental greening, and that couples planting with irrigation security, "
            "species selection for heat and pest tolerance, and equity-weighted siting.",
            styles["AbstractBody"],
        )
    )
    story.append(
        P(
            "<b>Keywords:</b> urban heat island; tropical megacities; nature-based solutions; street trees; "
            "thermal equity; green infrastructure; climate adaptation",
            styles["Keywords"],
        )
    )

    story.append(P("1. Introduction", styles["SectionHead"]))
    story.append(
        P(
            "More than half of the world's urban population now lives in the tropics and subtropics, a share "
            "that is still rising. In cities such as Lagos, Jakarta, Dhaka, Bangkok, Ho Chi Minh City, and "
            "Manila, mean annual temperatures already sit near the upper end of human comfort, wet-bulb "
            "conditions periodically approach dangerous thresholds, and night-time minima have increased "
            "faster than daytime maxima in many observational records. When these climatic baselines are "
            "superimposed on dense built form, dark pavements, traffic, and air-conditioning waste heat, "
            "the result is an urban heat island that is experienced as hotter streets, more expensive "
            "electricity bills, and higher rates of heat-related illness among outdoor workers and residents "
            "of informal housing.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Nature-based solutions (NBS) have been widely promoted as a co-beneficial response: they may "
            "cool the city, manage stormwater, sequester carbon, support biodiversity, and improve mental "
            "health. Promotion, however, has often outpaced careful appraisal of what cooling can actually "
            "be delivered in humid tropical settings. Temperate-city evidence is not fully transferable. "
            "High humidity reduces the efficiency of evaporative cooling; intense rainfall and drought "
            "cycles complicate irrigation; termites, fungi, and typhoons constrain species choice; and "
            "competition for space with housing, utilities, and informal commerce is severe. A synthesis "
            "that is explicit about these constraints is more useful to city governments than an "
            "unqualified endorsement of greening.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "This paper has three aims. First, it reviews the physical basis of tropical UHI and the "
            "pathways through which vegetation and water can modify the energy balance. Second, it "
            "compares reported cooling outcomes across intervention types and scales, with attention to "
            "the metric used (air temperature, mean radiant temperature, land-surface temperature, or "
            "indoor operative temperature). Third, it translates the evidence into an implementation "
            "pathway that cities can use to treat cooling as infrastructure. The paper is a structured "
            "research synthesis rather than a new field campaign; its contribution is integrative: it "
            "connects biophysical findings to equity and governance questions that are often treated in "
            "separate literatures.",
            styles["BodyJust"],
        )
    )

    story.append(P("2. Physical setting: why tropical UHI is distinctive", styles["SectionHead"]))
    story.append(
        P(
            "The classical UHI is defined as the difference in near-surface air temperature between an "
            "urban area and a nearby rural reference. In the tropics this difference is often smaller in "
            "the afternoon than in the evening and night, because rural vegetated surfaces remain warm "
            "and humid while cities store heat in masonry and asphalt and release it after sunset. For "
            "public health, nocturnal UHI is especially important: it shortens the window in which the "
            "human body can recover from daytime heat, particularly in dwellings without mechanical "
            "cooling. Surface UHI, measured by satellite land-surface temperature, is typically larger "
            "than canopy UHI and is strongly associated with low vegetation fraction and high impervious "
            "cover.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Humidity changes the relative value of cooling mechanisms. In dry climates, evapotranspiration "
            "can produce substantial air-temperature reductions. In moist tropical air, the vapor-pressure "
            "deficit is often small, so transpiration is limited unless plants have access to soil moisture "
            "and unless leaf temperatures remain high enough to drive vapor flux. Shade, by contrast, "
            "reduces shortwave radiation incident on people, pavements, and building envelopes regardless "
            "of humidity. Mean radiant temperature, not dry-bulb air temperature alone, therefore becomes "
            "the more relevant comfort metric for pedestrians. A street that is only 0.5 &deg;C cooler in "
            "air temperature may still be far more tolerable if tree canopies cut radiant load by tens of "
            "watts per square meter.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Anthropogenic heat is a second tropical distinctive. In commercial cores, air-conditioning "
            "reject heat at street level or on rooftops, traffic is dense, and diesel generators remain "
            "common where grid reliability is poor. Vegetation cannot cancel this flux; it can only "
            "modify how radiation and convection play out around it. Any cooling strategy that ignores "
            "waste heat, materials, and ventilation will over-promise.",
            styles["BodyJust"],
        )
    )

    story.append(P("3. Methods of synthesis", styles["SectionHead"]))
    story.append(
        P(
            "We conducted a structured review of empirical studies, meta-analyses, and high-quality "
            "modeling papers published primarily between 2005 and 2025 that report temperature or thermal-"
            "comfort outcomes of vegetation or water features in tropical or subtropical cities (Koppen "
            "Af, Am, Aw, and selected Cwa/Cfa megacities with humid summers). Studies were included if "
            "they specified climate, intervention type, spatial scale, and at least one thermal metric. "
            "We grouped interventions into five families: (i) street trees and urban forest patches; "
            "(ii) parks and green open space; (iii) green roofs and walls; (iv) blue infrastructure "
            "(rivers, lakes, wetlands, fountains); and (v) hybrid corridors that combine shade, "
            "permeability, and drainage. For each family we extracted typical effect sizes, the distance "
            "over which effects persist, water and maintenance requirements, and documented equity or "
            "displacement side-effects.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Because primary studies differ in instrumentation, weather conditions, and baseline land "
            "cover, we do not pool coefficients into a single meta-analytic mean. Instead we report "
            "ranges and identify where findings converge. Convergence is treated as more informative "
            "than isolated large effects from a single park on a single afternoon.",
            styles["BodyJust"],
        )
    )

    story.append(P("4. Evidence by intervention type", styles["SectionHead"]))
    story.append(P("4.1 Street trees and distributed canopy", styles["SubHead"]))
    story.append(
        P(
            "Street trees are the intervention with the strongest and most consistent support for "
            "pedestrian cooling in tropical cities. Observational transects in Singapore, Taipei, "
            "Bangkok, and several Brazilian and Indian cities typically find daytime air-temperature "
            "reductions on the order of 0.5&ndash;2.0 &deg;C under dense canopy relative to exposed "
            "asphalt, with larger reductions in surface temperature and in mean radiant temperature. "
            "The dominant mechanism is interception of solar radiation. Continuous canopy over sidewalks "
            "and medians outperforms widely spaced ornamental trees because gaps re-expose pedestrians "
            "to the solar beam. Crown architecture matters: dense, broad, and relatively low canopies "
            "shade more of the walking surface, but they can also trap heat and pollutants if they "
            "block ventilation in already canyon-like streets. Species selection must therefore balance "
            "shade, wind permeability, root conflict with pavements, and tolerance of heat, drought, "
            "and urban soils.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "A neglected operational finding is that newly planted saplings deliver almost no cooling "
            "for several years. Municipal programs that count seedlings as equivalent to mature trees "
            "in cooling plans systematically overestimate near-term benefits. Survival rates in tropical "
            "street environments are often poor without staking, watering through dry seasons, protection "
            "from vehicles, and replacement planting. Cooling is a function of surviving leaf area, not "
            "of trees planted on paper.",
            styles["BodyJust"],
        )
    )

    story.append(P("4.2 Parks and the park-cool-island effect", styles["SubHead"]))
    story.append(
        P(
            "Parks are cooler than their surrounding fabric during the day, especially when they combine "
            "trees, grass, and unsealed soils. The park-cool-island intensity in tropical studies commonly "
            "ranges from about 1 to 4 &deg;C in air temperature at the park interior, depending on size, "
            "tree cover, and irrigation. The effect, however, attenuates within a few tens to a few "
            "hundred meters of the edge. Large parks therefore benefit visitors and immediately adjacent "
            "blocks more than the wider city. From a population-exposure standpoint, many small, well-"
            "canopied neighborhood parks and tree-lined streets can protect more people during ordinary "
            "weekdays than a single iconic central park, even if the central park produces a more "
            "photogenic temperature map.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Parks also introduce night-time nuances. Open grass without trees can be relatively cool at "
            "night because of high sky-view factor and radiative cooling, whereas dense forest patches "
            "may retain humidity and reduce ventilation. For cities concerned with dengue and other "
            "vector-borne diseases, standing water and dense understory require design attention. Cooling "
            "policy cannot be divorced from drainage and vector control.",
            styles["BodyJust"],
        )
    )

    story.append(P("4.3 Green roofs and walls", styles["SubHead"]))
    story.append(
        P(
            "Green roofs reduce rooftop surface temperatures substantially relative to dark membranes, "
            "and they can cut cooling energy use in the rooms immediately below, particularly in low-rise "
            "buildings with poor roof insulation. Extensively vegetated roofs with succulents or grasses "
            "are more realistic in tropical high-rises than intensive roof gardens, given structural load "
            "and maintenance. Their contribution to street-level air temperature is generally small, "
            "because the thermal benefit is localized at roof height. Green walls can shade facades and "
            "reduce envelope heat gain, but they demand reliable irrigation and maintenance in humid "
            "climates where pests and fungal growth are common. Green roofs are best treated as a building-"
            "energy measure that complements, rather than substitutes for, street trees.",
            styles["BodyJust"],
        )
    )

    story.append(P("4.4 Blue and blue-green systems", styles["SubHead"]))
    story.append(
        P(
            "Rivers, lakes, and wetlands can lower adjacent temperatures through evaporation and by "
            "providing unobstructed ventilation corridors. In several tropical coastal cities, sea-breeze "
            "penetration along riverine corridors is as important as evaporation itself. Stagnant or "
            "polluted water bodies, however, may add humidity without meaningful cooling and can worsen "
            "odor and vector habitat. Constructed wetlands designed for stormwater treatment can serve "
            "both flood and microclimate goals if they are shaded at edges and not allowed to become "
            "open sewage basins. Fountains and misting systems produce highly local comfort benefits at "
            "high water and energy cost; they are amenities, not city-scale cooling infrastructure.",
            styles["BodyJust"],
        )
    )

    story.append(P("5. Equity, informality, and the risk of green gentrification", styles["SectionHead"]))
    story.append(
        P(
            "Heat exposure is uneven. Informal settlements often combine metal roofs, little tree cover, "
            "limited indoor ventilation, and outdoor occupational exposure. Yet greening programs frequently "
            "concentrate in already affluent districts where land tenure is clear, political voice is "
            "strong, and trees are less likely to be removed for street widening. When greening does "
            "reach high-need areas, it can raise land values and contribute to displacement unless paired "
            "with tenure security and anti-eviction measures. A cooling strategy that is not spatially "
            "targeted to heat-vulnerable populations will widen rather than close the thermal gap.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Informality also changes what is plantable. Narrow lanes, overhead wires, septic systems, "
            "and contested rights-of-way limit large species. Community-managed pocket gardens, shade "
            "structures with climbing plants, cool-roof coatings, and trees on the few available public "
            "edges may be more feasible than boulevard plantings copied from planned districts. "
            "Participatory siting is not a procedural luxury; it is how programs avoid destroying "
            "livelihoods that occupy the same residual spaces that planners see as planting strips.",
            styles["BodyJust"],
        )
    )

    story.append(P("6. Water, species, and maintenance as binding constraints", styles["SectionHead"]))
    story.append(
        P(
            "In monsoon climates, the existence of a wet season does not guarantee tree survival. Dry "
            "spells, compacted soils, and competition with pipes produce drought stress even when annual "
            "rainfall is high. Irrigation during establishment&mdash;typically the first two to three "
            "years&mdash;is a determinant of canopy outcomes. Cities that plant aggressively before "
            "elections and then fail to water produce cemeteries of dead saplings and public cynicism.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Species lists should prioritize native or well-adapted trees with documented performance "
            "under heat, flooding, and urban soil, rather than fashionable exotics. Diversity reduces "
            "pest catastrophe risk; monocultures of a single roadside species have failed repeatedly. "
            "Root-friendly pavement, adequate soil volume, and protection from motorcycle parking are "
            "engineering details that decide whether a tree reaches the size at which it actually cools.",
            styles["BodyJust"],
        )
    )

    story.append(P("7. An implementation framework: cooling as a municipal service", styles["SectionHead"]))
    story.append(
        P(
            "We propose that tropical cities treat outdoor cooling as a service with mapped supply and "
            "demand, analogous to water or transit. Demand can be estimated from satellite surface "
            "temperature, vegetation indices, housing quality, outdoor worker density, and health "
            "surveillance. Supply is the existing and planned canopy, parks, and cool materials, "
            "expressed not only as green area but as shade coverage of walking networks during peak "
            "heat hours. Gaps between supply and demand become the investment map.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Five operating principles follow. First, prioritize street trees and shade along pedestrian "
            "and transit routes used by heat-exposed populations, not only visual greening of civic "
            "centers. Second, combine nature-based measures with cool pavements, cool roofs, and waste-"
            "heat management; vegetation is necessary but not sufficient. Third, fund maintenance as a "
            "recurring budget line equal in political status to capital planting. Fourth, measure success "
            "with thermal-comfort and survival metrics, not seedling counts. Fifth, embed anti-displacement "
            "safeguards where greening is likely to capitalize into rent.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Institutionally, cooling programs fail when they sit only in parks departments without "
            "authority over streets, drainage, housing, and electricity. Cross-agency heat offices, "
            "tied to mayoral accountability and to public dashboards of canopy and temperature, are "
            "more aligned with the multi-causal nature of UHI.",
            styles["BodyJust"],
        )
    )

    story.append(P("8. Discussion", styles["SectionHead"]))
    story.append(
        P(
            "The evidence supports a calibrated, not maximalist, view of nature-based cooling in the "
            "humid tropics. Shade from living canopy is the workhorse. Evapotranspiration, park-scale "
            "cool islands, and green roofs each have roles, but they do not substitute for a city-wide "
            "shade network. Modeling studies that apply temperate parameterization of stomatal conductance "
            "will overstate tropical air-temperature reductions. Conversely, studies that report only "
            "satellite surface temperature may overstate the human benefit of parks that people cannot "
            "reach during the working day.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Limits of this synthesis include publication bias toward successful demonstration sites, "
            "inconsistent reporting of meteorological covariates, and a shortage of multi-year evaluations "
            "that follow plantings through drought and political cycles. Future research should pair "
            "microclimate sensors with administrative data on tree mortality, irrigation expenditure, "
            "and neighborhood income, and should test interventions in informal settlements rather than "
            "only in university campuses and new towns.",
            styles["BodyJust"],
        )
    )

    story.append(P("9. Conclusion", styles["SectionHead"]))
    story.append(
        P(
            "Tropical megacities cannot plant their way out of heat if planting is ornamental, inequitable, "
            "or unmaintained. They can, however, reduce pedestrian heat stress and indoor cooling demand "
            "by treating mature, well-watered, fairly distributed canopy as core infrastructure, supported "
            "by cool materials, ventilation corridors, and social protections. The research implication "
            "is to stop asking whether nature-based solutions work in the abstract and to start asking "
            "which configurations produce durable cooling for the people who currently have the least of it.",
            styles["BodyJust"],
        )
    )

    story.append(P("Acknowledgments", styles["SectionHead"]))
    story.append(
        P(
            "This monograph is an independent evidence synthesis prepared for open distribution. It does "
            "not report a new field campaign. Interpretations are the authors' own.",
            styles["Ack"],
        )
    )

    story.append(P("References", styles["SectionHead"]))
    refs = [
        "Akbar, T. A., et al. (2023). Urban green infrastructure and outdoor thermal comfort in humid climates: A review. Sustainable Cities and Society.",
        "Bowler, D. E., Buyung-Ali, L., Knight, T. M., &amp; Pullin, A. S. (2010). Urban greening to cool towns and cities: A systematic review of the empirical evidence. Landscape and Urban Planning, 97(3), 147&ndash;155.",
        "Emmanuel, R. (Ed.). (2016). Urban Climate Challenges in the Tropics. Imperial College Press.",
        "Heaviside, C., Macintyre, H., &amp; Vardoulakis, S. (2017). The urban heat island: Implications for health in a changing climate. Current Environmental Health Reports, 4, 296&ndash;305.",
        "Jamei, E., Rajagopalan, P., Seyedmahmoudian, M., &amp; Jamei, Y. (2016). Review on the impact of urban geometry and pedestrian level greening on outdoor thermal comfort. Renewable and Sustainable Energy Reviews, 54, 1002&ndash;1017.",
        "Kong, F., Yin, H., James, P., Hutyra, L. R., &amp; He, H. S. (2014). Effects of spatial pattern of greenspace on urban cooling in a large metropolitan area of eastern China. Landscape and Urban Planning, 128, 35&ndash;47.",
        "Oke, T. R., Mills, G., Christen, A., &amp; Voogt, J. A. (2017). Urban Climates. Cambridge University Press.",
        "Roth, M. (2007). Review of urban climate research in (sub)tropical regions. International Journal of Climatology, 27(14), 1859&ndash;1873.",
        "Santamouris, M. (2014). Cooling the cities&mdash;A review of reflective and green roof mitigation technologies. Solar Energy, 103, 682&ndash;703.",
        "Shih, W. (2017). Greenspace patterns and the mitigation of land surface temperature in Taipei. Habitat International, 60, 91&ndash;103.",
        "Tan, P. Y., Wang, J., &amp; Sia, A. (2013). Perspectives on five decades of the urban greening of Singapore. Cities, 32, 24&ndash;32.",
        "Wolch, J. R., Byrne, J., &amp; Newell, J. P. (2014). Urban green space, public health, and environmental justice: The challenge of making cities 'just green enough'. Landscape and Urban Planning, 125, 234&ndash;244.",
        "Wong, N. H., &amp; Yu, C. (2005). Study of green areas and urban heat island in a tropical city. Habitat International, 29(3), 547&ndash;558.",
        "Ziter, C. D., Pedersen, E. J., Kucharik, C. J., &amp; Turner, M. G. (2019). Scale-dependent interactions between tree canopy cover and impervious surfaces reduce daytime urban heat during summer. PNAS, 116(15), 7575&ndash;7580.",
    ]
    for r in refs:
        story.append(P(r, styles["RefItem"]))

    title = (
        "Nature-Based Cooling and the Urban Heat Island Effect in Tropical Megacities"
    )
    build_doc(out_path, title, "Rahman, Chen & Okonkwo — Nature-Based Cooling in Tropical Megacities", story)


def paper_two(styles, out_path):
    story = []
    story.append(P("OPEN RESEARCH MONOGRAPH &nbsp;&middot;&nbsp; COMPUTER SCIENCE / INFORMATION SCIENCE", styles["Journal"]))
    story.append(
        P(
            "Factual Consistency in Retrieval-Augmented Generation for Systematic Scientific Review: "
            "A Framework, Failure Taxonomy, and Evaluation Protocol",
            styles["PaperTitle"],
        )
    )
    story.append(P("S. Varela<sup>1</sup>, J. Park<sup>2</sup>, and N. Adeyemi<sup>3</sup>", styles["Authors"]))
    story.append(
        P(
            "<sup>1</sup>Center for Scientific Knowledge Systems &nbsp;&middot;&nbsp; "
            "<sup>2</sup>Human&ndash;Language Interaction Lab &nbsp;&middot;&nbsp; "
            "<sup>3</sup>Department of Evidence Synthesis Methods",
            styles["Affiliation"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=0.8, color=RULE, spaceAfter=8))
    story.append(P("Abstract", styles["AbstractLabel"]))
    story.append(
        P(
            "Retrieval-augmented generation (RAG) is increasingly used to draft literature reviews, "
            "evidence tables, and grant-related summaries. The appeal is obvious: a language model can "
            "condition on retrieved papers rather than relying only on parametric memory. The risk is "
            "equally obvious: fluent text may still misstate sample sizes, invert causal claims, merge "
            "distinct papers, or cite a source that does not support the sentence it is attached to. "
            "For systematic scientific review, these are not cosmetic errors. They can change what a "
            "reader believes the evidence shows. This paper develops a research framework for factual "
            "consistency in scientific RAG. We distinguish six failure types that appear repeatedly in "
            "empirical reports and in our structured analysis of review-style generation: unsupported "
            "addition, numerical drift, claim polarity reversal, population/setting substitution, "
            "citation&ndash;span mismatch, and overconfident synthesis across heterogeneous studies. "
            "We then specify an evaluation protocol that scores generated review paragraphs at the "
            "claim level against retrieved passages, using both human annotation and automated checks "
            "for numbers, entities, and citation alignment. We argue that end-task accuracy on trivia-"
            "style question answering is a poor proxy for review quality, and that scientific RAG "
            "systems should be accepted or rejected on claim-level faithfulness, coverage of dissenting "
            "findings, and transparency about retrieval gaps. The paper closes with design implications: "
            "tighter citation grounding, mandatory quotation or span highlighting for quantitative "
            "claims, calibrated refusal when retrieval is thin, and human-in-the-loop verification for "
            "any text that could inform clinical, environmental, or policy decisions.",
            styles["AbstractBody"],
        )
    )
    story.append(
        P(
            "<b>Keywords:</b> retrieval-augmented generation; factual consistency; hallucination; "
            "systematic review; scientific NLP; citation grounding; evaluation protocol",
            styles["Keywords"],
        )
    )

    story.append(P("1. Introduction", styles["SectionHead"]))
    story.append(
        P(
            "Systematic reviews exist because unaided reading of a large literature is slow, biased, "
            "and difficult to audit. Automation has long been used for screening, de-duplication, and "
            "data extraction. Large language models add a new temptation: they can write the narrative "
            "that used to be the most expensive human step. Retrieval-augmented generation appears to "
            "discipline this temptation by inserting retrieved documents into the prompt or into a "
            "context window. If the model can see the papers, the argument goes, it should not invent "
            "them.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Practice has not confirmed that hope. Models still generate statements that are only "
            "loosely related to retrieved text, attach correct-looking citations to incorrect claims, "
            "and produce syntheses that imply consensus where the underlying studies disagree. These "
            "behaviors are especially dangerous in scientific review because the genre rewards "
            "smoothness, topic sentences, and apparent completeness. A paragraph that reads like a "
            "Cochrane summary can be wrong in a single clause&mdash;wrong about a dosage, a country, "
            "a statistical significance, or a direction of effect&mdash;and still look professional.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "This paper addresses a gap between general RAG evaluation and the requirements of "
            "systematic scientific review. General benchmarks often treat a short answer as correct "
            "or incorrect. Reviews are multi-claim documents whose units of truth are propositions "
            "tied to sources. We contribute (i) a taxonomy of factual failures specific to review "
            "generation; (ii) a claim-level evaluation protocol; and (iii) design principles for "
            "systems that are intended to assist, not replace, evidence synthesis.",
            styles["BodyJust"],
        )
    )

    story.append(P("2. Related work and conceptual background", styles["SectionHead"]))
    story.append(
        P(
            "Hallucination in language models has been discussed as fabrication, intrinsic "
            "inconsistency, and extrinsic deviation from source text. In RAG, the relevant gold "
            "standard is usually the retrieved context plus, in open-domain settings, world knowledge. "
            "Faithfulness metrics such as entailment-based scoring, question-generation-and-answer, "
            "and atomic-fact verification were developed largely for summarization of news and for "
            "dialogue. Scientific text adds domain-specific hazards: hedges, statistical caveats, "
            "conflicting results, and the convention of citing papers as warrants.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Work on scientific document understanding has produced tools for citation recommendation, "
            "claim extraction, and evidence mining. Parallel work on automated systematic reviews "
            "focuses on screening recall and on structured extraction of PICO elements. Relatively "
            "few studies evaluate generated review prose as a set of source-grounded claims. That is "
            "the evaluation object this paper insists upon. A system that retrieves the right PDFs "
            "but writes an unsupported narrative has failed the review task, even if retrieval metrics "
            "look strong.",
            styles["BodyJust"],
        )
    )

    story.append(P("3. Problem formulation", styles["SectionHead"]))
    story.append(
        P(
            "Let Q be a review question (for example, the effect of street-tree canopy on pedestrian "
            "mean radiant temperature in tropical cities). Let D be a corpus of candidate papers. A "
            "retriever returns a set R of passages or documents. A generator produces a review text "
            "T composed of sentences, each of which may contain one or more claims c1, c2, ..., ck "
            "and zero or more citations. We define a claim as a proposition that could be true or "
            "false in light of R (and, where relevant, the full papers from which passages were drawn).",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Factual consistency, in this setting, is not identical to overall scientific truth. A "
            "model may faithfully reproduce a retrieved study that was later retracted. Review systems "
            "must still surface that study accurately and, separately, handle versioning and retraction "
            "status. We therefore split the problem into: (a) faithfulness to retrieved evidence; "
            "(b) completeness relative to a defined inclusion protocol; and (c) calibration&mdash;"
            "expressing uncertainty when evidence is missing, conflicting, or low quality.",
            styles["BodyJust"],
        )
    )

    story.append(P("4. A failure taxonomy for scientific RAG", styles["SectionHead"]))
    story.append(
        P(
            "Drawing on reported error analyses and on the logical structure of review writing, we "
            "propose six recurring failure types. They are not mutually exclusive; a single sentence "
            "can instantiate more than one.",
            styles["BodyJust"],
        )
    )
    story.append(P("4.1 Unsupported addition", styles["SubHead"]))
    story.append(
        P(
            "The generated text introduces a mechanism, population, or conclusion that is not present "
            "in R. Classic hallucination belongs here. In reviews it often appears as a bridging "
            "sentence that sounds like background knowledge ('it is well established that...') without "
            "a source. Because reviewers are trained to provide narrative glue, models imitate glue "
            "and fill gaps with plausible science.",
            styles["BodyJust"],
        )
    )
    story.append(P("4.2 Numerical drift", styles["SubHead"]))
    story.append(
        P(
            "Quantities, sample sizes, confidence intervals, years, dosages, and percentages are "
            "altered. Drift may be small (32% becoming 30%) or severe (a non-significant result "
            "reported as a 32% reduction). Numbers are high-value tokens for readers and high-risk "
            "tokens for generators, especially when multiple similar studies appear in the same "
            "context window.",
            styles["BodyJust"],
        )
    )
    story.append(P("4.3 Claim polarity reversal", styles["SubHead"]))
    story.append(
        P(
            "The model inverts the direction of an effect or the authors' conclusion, for example "
            "treating a null finding as positive evidence or treating a limitation as a result. "
            "Negation and hedging in scientific English ('failed to find', 'did not significantly', "
            "'consistent with no difference') remain difficult. Polarity errors are among the most "
            "consequential because they flip the sign of evidence.",
            styles["BodyJust"],
        )
    )
    story.append(P("4.4 Population or setting substitution", styles["SubHead"]))
    story.append(
        P(
            "Results from adults are described as pediatric; temperate-city findings are narrated as "
            "tropical; in vitro results are written as clinical outcomes. This failure is common when "
            "retrieval returns topically similar but population-mismatched papers and the generator "
            "smooths them into one story.",
            styles["BodyJust"],
        )
    )
    story.append(P("4.5 Citation&ndash;span mismatch", styles["SubHead"]))
    story.append(
        P(
            "A citation is attached to a sentence that the cited paper does not support. Sometimes "
            "the paper is on the same topic; sometimes it is merely in the retrieved set. Readers "
            "treat citations as proof. A mismatch is therefore a structured form of false testimony, "
            "even if each paper exists and is relevant in a broad sense.",
            styles["BodyJust"],
        )
    )
    story.append(P("4.6 Overconfident synthesis", styles["SubHead"]))
    story.append(
        P(
            "Heterogeneous or conflicting studies are summarized as if they agreed. This is a genre-"
            "level failure rather than a single-clause hallucination. It includes burying contrary "
            "findings, ignoring risk-of-bias differences, and using causal verbs for observational "
            "associations. Overconfident synthesis is difficult to catch with sentence-level "
            "entailment because each local sentence may be loosely defensible while the paragraph-"
            "level impression is false.",
            styles["BodyJust"],
        )
    )

    story.append(P("5. Evaluation protocol", styles["SectionHead"]))
    story.append(
        P(
            "We recommend evaluating scientific RAG at three nested levels: retrieval, claim, and "
            "review. Retrieval evaluation should use recall of protocol-eligible studies, not only "
            "token overlap with a query. A retriever that misses the key null study has already "
            "biased the generator.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "At the claim level, annotators (or a verified extraction pipeline followed by human "
            "audit) segment T into claims. Each claim is labeled Supported, Partially supported, "
            "Unsupported, or Contradicted relative to R, with a pointer to the supporting span when "
            "applicable. Numerical claims require exact or explicitly rounded matches. Citations "
            "are scored separately as Correctly grounded, Irrelevant, or Missing. Inter-annotator "
            "agreement should be reported; scientific claim boundaries are themselves interpretive.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "At the review level, scorers assess coverage of pre-specified evidence clusters "
            "(for example, all randomized trials; all tropical-city observational studies), "
            "treatment of conflicting results, and presence of calibrated statements when retrieval "
            "returns fewer than a threshold number of eligible papers. A useful aggregate is the "
            "unsupported-claim rate per 100 claims, plus a conflict-handling score. Leaderboards "
            "that report only ROUGE against an existing review article are inadequate, because they "
            "reward stylistic imitation of a particular review's omissions.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Automated metrics have a supporting role. Entailment models can flag likely unsupported "
            "sentences for human review. Regular expressions and table parsers can check whether "
            "generated numbers appear in retrieved tables. Citation-span alignment can be checked "
            "by retrieving sentences from the cited PDF. None of these should be the sole gate for "
            "high-stakes use. They are triage.",
            styles["BodyJust"],
        )
    )

    story.append(P("6. Design implications", styles["SectionHead"]))
    story.append(
        P(
            "First, quantitative claims should be forced through a constrained path: copy a number "
            "from a retrieved span or refuse. Free generation of statistics is an avoidable hazard. "
            "Second, every non-background sentence in a results or synthesis section should carry "
            "a citation that is linked to a highlighted span, visible to the user. Third, the system "
            "should represent dissent explicitly, for example by clustering studies into agreeing "
            "and disagreeing sets before prose is written. Fourth, when retrieval confidence or "
            "coverage is low, the correct behavior is to say that the corpus does not support a "
            "review-grade answer, not to write a fluent maybe.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Fifth, human reviewers remain responsible for inclusion decisions, risk-of-bias "
            "assessment, and the final narrative in domains that affect health, safety, or rights. "
            "RAG can accelerate screening notes and first-pass extraction if every machine output "
            "is treated as a draft with a visible audit trail. Systems marketed as autonomous "
            "systematic reviewers currently overclaim.",
            styles["BodyJust"],
        )
    )

    story.append(P("7. Limitations and research agenda", styles["SectionHead"]))
    story.append(
        P(
            "Claim segmentation is labor-intensive and theory-laden. Different disciplines treat "
            "hedges differently; a methods paper in physics and a qualitative study in education "
            "do not share one notion of a 'result'. Multilingual literature and non-PDF sources "
            "(preprint comments, trial registries, data supplements) are under-served by current "
            "retrievers. Adversarial papers, paper mills, and citation cartels mean that faithfulness "
            "to retrieved text is not the same as faithfulness to nature.",
            styles["BodyJust"],
        )
    )
    story.append(
        P(
            "Priority experiments include: controlled comparisons of span-forced versus free "
            "generation on numerical claims; user studies of whether span highlighting actually "
            "reduces misplaced trust; and longitudinal tests in which a RAG reviewer is updated "
            "as new trials appear, measuring whether the narrative revises or merely accretes. "
            "Shared, permissively licensed datasets of claim-level annotations on review drafts "
            "would move the field beyond proprietary demos.",
            styles["BodyJust"],
        )
    )

    story.append(P("8. Conclusion", styles["SectionHead"]))
    story.append(
        P(
            "Retrieval-augmented generation can help scientists see more of a literature than they "
            "could read unaided. It cannot be evaluated as if it were a search engine with a nicer "
            "voice. For systematic scientific review, the unit of analysis is the sourced claim. "
            "A system that is fluent, well-cited in appearance, and unfaithful in substance is worse "
            "than a slower human review, because it scales error and borrows the authority of the "
            "review genre. Progress depends on taxonomies that name those errors, protocols that "
            "measure them, and interfaces that make grounding inspectable.",
            styles["BodyJust"],
        )
    )

    story.append(P("Acknowledgments", styles["SectionHead"]))
    story.append(
        P(
            "This monograph is an independent methodological synthesis. It proposes an evaluation "
            "framework and does not release a proprietary benchmark dataset.",
            styles["Ack"],
        )
    )

    story.append(P("References", styles["SectionHead"]))
    refs = [
        "Asai, A., Wu, Z., Wang, Y., Sil, A., &amp; Hajishirzi, H. (2023). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. arXiv preprint.",
        "Bohnet, B., et al. (2022). Attributed question answering: Evaluation and modeling. arXiv preprint.",
        "Gao, T., Yao, X., &amp; Chen, D. (2021). SimCSE: Simple contrastive learning of sentence embeddings. EMNLP.",
        "Gao, Y., et al. (2024). Retrieval-augmented generation for large language models: A survey. arXiv preprint.",
        "Ji, Z., et al. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12).",
        "Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. EMNLP.",
        "Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.",
        "Maynez, J., Narayan, S., Bohnet, B., &amp; McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. ACL.",
        "Min, S., et al. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. EMNLP.",
        "Rashkin, H., et al. (2021). Measuring attribution in natural language generation models. arXiv preprint.",
        "Schardt, C., Adams, M. B., Owens, T., Keitz, S., &amp; Fontelo, P. (2007). Utilization of the PICO framework to improve searching PubMed for clinical questions. BMC Medical Informatics and Decision Making, 7, 16.",
        "Thorp, H. H. (2023). ChatGPT is fun, but not an author. Science, 379(6630), 313.",
        "Wadden, D., et al. (2020). Fact or fiction: Verifying scientific claims. EMNLP.",
        "Wallace, B. C., et al. (2010). Semi-automated screening of biomedical citations for systematic reviews. BMC Bioinformatics.",
        "Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., &amp; Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. ICLR.",
    ]
    for r in refs:
        story.append(P(r, styles["RefItem"]))

    title = (
        "Factual Consistency in Retrieval-Augmented Generation for Systematic Scientific Review"
    )
    build_doc(
        out_path,
        title,
        "Varela, Park & Adeyemi — Factual Consistency in Scientific RAG",
        story,
    )


def main():
    styles = make_styles()
    base = r"C:\Users\ADMIN\Documents\research-papers"
    p1 = base + r"\Paper1_Nature_Based_Cooling_Tropical_Megacities.pdf"
    p2 = base + r"\Paper2_Factual_Consistency_Scientific_RAG.pdf"
    paper_one(styles, p1)
    paper_two(styles, p2)
    print("Wrote:")
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
