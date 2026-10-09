"""
Attribute selector nodes: Female Person, Female Fashion, Female Actions, Scene.

Every dropdown offers "none" (left out of the prompt), "random" (resolved by the
generator node from its seed) or a specific value. The nodes are chained through
the KREA_ATTRIBUTES type and finally plugged into the Krea Prompt Generator.
"""

import json
import os

NONE = "none"
RANDOM = "random"
ATTRIBUTES_TYPE = "KREA_ATTRIBUTES"
CATEGORY = "KreaPromptGenerator"

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "attributes.json")

_data_cache = None


def load_data():
    global _data_cache
    if _data_cache is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # The source lists use "-" as a placeholder; "none" replaces it here.
        _data_cache = {
            key: [v for v in values if isinstance(v, str) and v.strip() not in ("", "-")]
            for key, values in raw.items()
            if isinstance(values, list)
        }
    return _data_cache


# Values that are not in the JSON file.
EXTRA_LISTS = {
    "age_list": [
        "18-21 years old", "22-25 years old", "26-30 years old", "31-35 years old",
        "36-40 years old", "41-45 years old", "46-50 years old", "51-55 years old",
        "56-60 years old", "60+ years old",
    ],
    "time_of_day_list": [
        "Dawn", "Morning", "Midday", "Afternoon", "Golden Hour", "Sunset",
        "Dusk", "Evening", "Night", "Midnight", "Blue Hour",
    ],
    "weather_list": [
        "Sunny", "Cloudy", "Overcast", "Rainy", "Stormy", "Snowy",
        "Foggy", "Misty", "Windy", "Clear",
    ],
    "season_list": ["Spring", "Summer", "Autumn", "Winter"],
}


def options_for(list_keys):
    """Merge one or more data lists into a de-duplicated option list."""
    data = load_data()
    seen, merged = set(), []
    for key in list_keys:
        for value in EXTRA_LISTS.get(key) or data.get(key, []):
            if value not in seen:
                seen.add(value)
                merged.append(value)
    return merged


# Each field: (input name, label shown to the LLM, list keys in the data file)
FEMALE_PERSON_FIELDS = [
    ("age", "Age", ["age_list"]),
    ("nationality", "Nationality / ethnicity", ["nationality_list"]),
    ("body_type", "Body type", ["body_type_list"]),
    ("height", "Height", ["height_list"]),
    ("body_weight", "Body weight", ["body_weight_list"]),
    ("waist_size", "Waist", ["waist_size_list"]),
    ("breast_shape", "Breast shape", ["breast_shape_list"]),
    ("breast_size", "Breast size", ["breast_size_list"]),
    ("breast_cup_size", "Bra cup size", ["breast_cup_size_list"]),
    ("bum_size", "Buttocks", ["bum_size_list"]),
    ("face_shape", "Face shape", ["face_shape_list"]),
    ("eyes_color", "Eye color", ["eyes_color_list"]),
    ("eye_shape", "Eye shape", ["eye_shape_list"]),
    ("eyes_details", "Eye detail", ["eyes_details_list"]),
    ("iris_details", "Iris detail", ["iris_details_list"]),
    ("circular_iris", "Iris shape", ["circular_iris_list"]),
    ("circular_pupil", "Pupils", ["circular_pupil_list"]),
    ("nose_shape", "Nose shape", ["nose_shape_list"]),
    ("nose_size", "Nose size", ["nose_size_list"]),
    ("lip_shape", "Lips", ["lip_shape_list"]),
    ("lip_color", "Lip color", ["lip_color_list"]),
    ("makeup", "Makeup", ["makeup_list"]),
    ("facial_expression", "Facial expression", ["face_expression_list"]),
    ("hair_style", "Hair style", ["hair_style_list"]),
    ("hair_length", "Hair length", ["hair_length_list"]),
    ("hair_color", "Hair color", ["hair_color_list"]),
    ("skin_tone", "Skin tone", ["skin_tone_list"]),
    ("skin_details", "Skin texture", ["skin_details_list"]),
    ("skin_pores", "Skin pores", ["skin_pores_list"]),
    ("tanned_skin", "Tan", ["tanned_skin_list"]),
    ("freckles", "Freckles", ["freckles_list"]),
    ("dimples", "Dimples", ["dimples_list"]),
    ("moles", "Moles", ["moles_list"]),
    ("skin_imperfections", "Skin imperfections", ["skin_imperfections_list"]),
    ("skin_acne", "Blemishes", ["skin_acne_list"]),
    ("nipple_appearance", "Nipples", ["nipple_appearance_list"]),
    ("areola_appearance", "Areolas", ["areola_appearance_list"]),
    ("tattoo", "Tattoos", ["tattoo_list"]),
]

FEMALE_FASHION_FIELDS = [
    ("fashion_aesthetic", "Overall fashion aesthetic", ["fashion_aesthetic_list"]),
    ("tops", "Top", ["tops_list"]),
    ("tops_color", "Top color", ["tops_color_list"]),
    ("tops_material", "Top material", ["tops_material_list"]),
    ("pants", "Bottoms (pants / skirt / shorts)", ["pants_list"]),
    ("pants_color", "Bottoms color", ["pants_color_list"]),
    ("pants_material", "Bottoms material", ["pants_material_list"]),
    ("dresses", "Dress", ["dresses_list"]),
    ("dresses_color", "Dress color", ["dresses_color_list"]),
    ("dresses_material", "Dress material", ["dresses_material_list"]),
    ("womens_suits", "Costume / suit", ["womens_suits_list"]),
    ("womens_suits_primary_color", "Costume primary color", ["womens_suits_primary_color_list"]),
    ("womens_suits_accent_color", "Costume accent color", ["womens_suits_accent_color_list"]),
    ("underwear", "Underwear / lingerie", ["underwear_list"]),
    ("underwear_color", "Underwear color", ["underwear_color_list"]),
    ("underwear_material", "Underwear material", ["underwear_material_list"]),
    ("legs", "Legwear", ["legs_list"]),
    ("legs_color", "Legwear color", ["legs_color_list"]),
    ("legs_material", "Legwear material", ["legs_material_list"]),
    ("capes", "Outerwear / cape", ["capes_list"]),
    ("capes_color", "Outerwear color", ["capes_color_list"]),
    ("capes_material", "Outerwear material", ["capes_material_list"]),
    ("hats", "Hat", ["hats_list"]),
    ("hats_color", "Hat color", ["hats_color_list"]),
    ("womens_shoes", "Shoes", ["womens_shoes_list"]),
    ("womens_shoe_color", "Shoe color", ["womens_shoe_color_list"]),
    ("womens_shoe_material", "Shoe material", ["womens_shoe_material_list"]),
    ("womens_gloves", "Gloves", ["womens_gloves_list"]),
    ("womens_gloves_color", "Glove color", ["womens_gloves_color_list"]),
    ("womens_gloves_material", "Glove material", ["womens_gloves_material_list"]),
    ("womens_belt", "Belt", ["womens_belt_list"]),
    ("womens_belt_color", "Belt color", ["womens_belt_color_list"]),
    ("womens_belt_material", "Belt material", ["womens_belt_material_list"]),
    ("womens_glasses", "Glasses", ["womens_glasses_list"]),
    ("womens_glasses_color", "Glasses color", ["womens_glasses_color_list"]),
    ("womens_mask", "Mask", ["womens_mask_list"]),
    ("womens_mask_color", "Mask color", ["womens_mask_color_list"]),
    ("womens_mask_material", "Mask material", ["womens_mask_material_list"]),
    ("necklace", "Necklace", ["necklace_list"]),
    ("earrings", "Earrings", ["earrings_list"]),
    ("bracelet", "Bracelet", ["bracelet_list"]),
    ("ring", "Ring", ["ring_list"]),
    ("watches", "Watch", ["watches_list"]),
    ("watches_color", "Watch color", ["watches_color_list"]),
    ("fingernail_style", "Nails", ["fingernail_style_list"]),
    ("nail_color", "Nail color", ["nail_color_list"]),
]

FEMALE_ACTIONS_FIELDS = [
    # One pose dropdown per type so "random" on several of them can conflict;
    # the generator is told to keep only the first pose if that happens.
    ("standing_pose", "Pose (standing)", ["standing_pose_list"]),
    ("sitting_pose", "Pose (sitting)", ["sitting_pose_list"]),
    ("kneeling_pose", "Pose (kneeling)", ["kneeling_pose_list"]),
    ("laying_down_pose", "Pose (lying down)", ["laying_down_pose_list"]),
    ("hand_placement", "Hands", ["hand_placement_list"]),
    ("leg_placement", "Legs", ["leg_placement_list"]),
    ("head_placement", "Head", ["head_placement_list"]),
    ("props", "Prop / action", ["props_list"]),
    ("props_color", "Prop color", ["props_color_list"]),
]

SCENE_FIELDS = [
    ("artistic_style", "Medium / artistic style", ["artistic_style_list"]),
    ("location", "Location", ["location_list"]),
    ("time_of_day", "Time of day", ["time_of_day_list"]),
    ("season", "Season", ["season_list"]),
    ("weather", "Weather", ["weather_list"]),
    ("light_type", "Lighting", ["light_type_list"]),
    ("light_quality", "Light quality", ["light_quality_list"]),
    ("camera_shot", "Camera shot / framing", ["camera_shot_list"]),
    ("camera_view", "Camera view", ["camera_view_list"]),
    ("camera_horizontal_angle", "Camera horizontal angle", ["camera_horizontal_angle_list"]),
    ("camera_vertical_angle", "Camera vertical angle", ["camera_vertical_angle_list"]),
    ("camera_lens", "Lens", ["camera_lens_specs"]),
    ("camera_model", "Camera", ["camera_model_list"]),
]


def make_attribute_node(class_name, group, fields, custom_field):
    """Build a selector node class from a field table."""

    class _Node:
        GROUP = group
        FIELDS = fields

        @classmethod
        def INPUT_TYPES(cls):
            required = {}
            for name, _label, list_keys in cls.FIELDS:
                required[name] = ([NONE, RANDOM] + options_for(list_keys), {"default": NONE})
            optional = {
                "attributes": (ATTRIBUTES_TYPE, {"tooltip": "Chain from another attribute node"}),
                custom_field: (
                    "STRING",
                    {"multiline": True, "default": "", "placeholder": f"Extra {group.lower()} details (optional)"},
                ),
            }
            return {"required": required, "optional": optional}

        RETURN_TYPES = (ATTRIBUTES_TYPE,)
        RETURN_NAMES = ("attributes",)
        FUNCTION = "run"
        CATEGORY = CATEGORY

        def run(self, attributes=None, **kwargs):
            entries = list(attributes) if attributes else []
            # Replace earlier entries from the same group so re-chaining a node type overrides it.
            entries = [e for e in entries if e["group"] != self.GROUP]
            for name, label, list_keys in self.FIELDS:
                value = kwargs.get(name, NONE)
                if value == NONE:
                    continue
                entries.append(
                    {
                        "group": self.GROUP,
                        "field": name,
                        "label": label,
                        "value": value,
                        # Carried along so the generator can resolve "random".
                        "options": options_for(list_keys) if value == RANDOM else None,
                    }
                )
            custom = (kwargs.get(custom_field) or "").strip()
            if custom:
                entries.append(
                    {"group": self.GROUP, "field": custom_field, "label": "Additional details", "value": custom, "options": None}
                )
            return (entries,)

    _Node.__name__ = class_name
    _Node.__qualname__ = class_name
    return _Node


KreaFemalePerson = make_attribute_node("KreaFemalePerson", "Physical characteristics", FEMALE_PERSON_FIELDS, "custom_person")
KreaFemaleFashion = make_attribute_node("KreaFemaleFashion", "Fashion", FEMALE_FASHION_FIELDS, "custom_fashion")
KreaFemaleActions = make_attribute_node("KreaFemaleActions", "Pose and action", FEMALE_ACTIONS_FIELDS, "custom_action")
KreaScene = make_attribute_node("KreaScene", "Scene, lighting and camera", SCENE_FIELDS, "custom_scene")

NODE_CLASS_MAPPINGS = {
    "KreaFemalePerson": KreaFemalePerson,
    "KreaFemaleFashion": KreaFemaleFashion,
    "KreaFemaleActions": KreaFemaleActions,
    "KreaScene": KreaScene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaFemalePerson": "Krea Prompt - Female Person",
    "KreaFemaleFashion": "Krea Prompt - Female Fashion",
    "KreaFemaleActions": "Krea Prompt - Female Actions",
    "KreaScene": "Krea Prompt - Scene",
}
