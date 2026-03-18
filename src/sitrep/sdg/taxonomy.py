from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SdgDefinition:
    number: int
    name: str
    description: str
    keywords: tuple[str, ...]


SDG_DEFINITIONS: tuple[SdgDefinition, ...] = (
    SdgDefinition(1, "No Poverty", "Eradicate poverty in all its forms globally, with a focus on ensuring basic human needs such as food, shelter, and clean water, and increasing access to social protection and economic resources.", ("poverty", "cash", "livelihood", "income", "social protection", "basic needs", "shelter", "economic resources")),
    SdgDefinition(2, "Zero Hunger", "End hunger and malnutrition by promoting sustainable agriculture, food security, improved nutrition, and equitable access to sufficient, nutritious food year-round.", ("hunger", "food security", "nutrition", "malnutrition", "famine", "agriculture", "harvest", "crop")),
    SdgDefinition(3, "Good Health and Well-being", "Ensure healthy lives and well-being for all by reducing maternal and child mortality, ending epidemics, improving healthcare systems, and ensuring universal access to health services.", ("health", "disease", "epidemic", "cholera", "hospital", "clinic", "vaccination", "medical", "well-being")),
    SdgDefinition(4, "Quality Education", "Provide inclusive, equitable, and high-quality education for all and promote lifelong learning opportunities to ensure literacy, numeracy, and access to skills for sustainable development.", ("education", "school", "teacher", "student", "learning", "literacy", "classroom")),
    SdgDefinition(5, "Gender Equality", "Achieve gender equality by eliminating all forms of discrimination, violence, and harmful practices against women and girls, and empowering them through equal opportunities in leadership, education, and the workforce.", ("gender", "women", "girls", "gbv", "sexual violence", "female", "empowerment")),
    SdgDefinition(6, "Clean Water and Sanitation", "Ensure universal access to clean water and adequate sanitation by improving water quality, managing water resources, reducing pollution, and promoting sustainable practices.", ("water", "sanitation", "wash", "hygiene", "latrine", "drinking water", "wastewater")),
    SdgDefinition(7, "Affordable and Clean Energy", "Ensure universal access to affordable, reliable, and modern energy by increasing renewable energy production, improving energy efficiency, and promoting sustainable energy consumption.", ("energy", "electricity", "power", "fuel", "solar", "generator", "renewable")),
    SdgDefinition(8, "Decent Work and Economic Growth", "Promote sustained, inclusive, and sustainable economic growth by providing productive employment, protecting labor rights, ensuring decent work conditions, and promoting economic equality.", ("employment", "jobs", "work", "labor", "wages", "market", "economy", "income generation")),
    SdgDefinition(9, "Industry, Innovation, and Infrastructure", "Build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation to drive economic growth, technological progress, and sustainable development.", ("infrastructure", "road", "bridge", "telecom", "innovation", "industry", "supply chain", "logistics")),
    SdgDefinition(10, "Reduced Inequality", "Reduce income inequality within and among countries by promoting social, economic, and political inclusion, as well as equal opportunities for marginalized and disadvantaged populations.", ("inequality", "marginalized", "inclusion", "discrimination", "minority", "disability", "equity")),
    SdgDefinition(11, "Sustainable Cities and Communities", "Make cities and human settlements inclusive, safe, resilient, and sustainable by ensuring affordable housing, reducing urban pollution, improving infrastructure, and promoting sustainable urban planning.", ("urban", "city", "housing", "settlement", "community", "municipal", "camp planning")),
    SdgDefinition(12, "Responsible Consumption and Production", "Ensure sustainable consumption and production patterns by reducing waste, increasing recycling, promoting sustainable business practices, and encouraging responsible consumer behavior.", ("waste", "recycling", "consumption", "production", "supply", "procurement", "pollution")),
    SdgDefinition(13, "Climate Action", "Take urgent action to combat climate change and its impacts by reducing greenhouse gas emissions, building resilience to climate-related hazards, and integrating climate policies into national development strategies.", ("climate", "flood", "drought", "heat", "resilience", "weather", "adaptation", "mitigation")),
    SdgDefinition(14, "Life Below Water", "Conserve and sustainably use the oceans, seas, and marine resources by reducing marine pollution, protecting coastal ecosystems, and promoting sustainable fishing practices.", ("ocean", "sea", "marine", "coastal", "fishery", "fishing", "coral")),
    SdgDefinition(15, "Life on Land", "Protect, restore, and promote sustainable use of terrestrial ecosystems, manage forests sustainably, combat desertification, halt biodiversity loss, and protect natural habitats.", ("forest", "biodiversity", "land", "wildlife", "desertification", "ecosystem", "pasture", "crop land")),
    SdgDefinition(16, "Peace, Justice, and Strong Institutions", "Promote peaceful and inclusive societies by ensuring access to justice, reducing violence, building accountable institutions, and fostering good governance at all levels.", ("violence", "conflict", "protection", "justice", "governance", "institution", "detention", "human rights")),
    SdgDefinition(17, "Partnerships for the Goals", "Strengthen global partnerships for sustainable development by mobilizing resources, sharing knowledge and technology, and promoting international cooperation for achieving the SDGs.", ("partnership", "coordination", "donor", "funding", "cooperation", "collaboration", "capacity building")),
)


def ordered_sdg_names() -> list[str]:
    return [definition.name for definition in SDG_DEFINITIONS]


def sdg_descriptions_for_prompt() -> list[dict[str, object]]:
    return [
        {
            "number": definition.number,
            "name": definition.name,
            "description": definition.description,
        }
        for definition in SDG_DEFINITIONS
    ]
