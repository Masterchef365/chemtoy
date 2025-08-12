use std::collections::HashMap;

use egui::DragValue;
use laws::{ChemicalWorld, Compound, Element, Elements, Laws};

mod laws;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                // NOTE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "eframe template",
        native_options,
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
    )
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}

pub struct TemplateApp {
    //chem: ChemicalWorld,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct SaveData {
    example_value: f32,
}

impl Default for SaveData {
    fn default() -> Self {
        Self {
            // Example stuff:
            example_value: 2.7,
        }
    }
}

impl TemplateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        /*
        let save_data = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();
        */

        /*
        let mut elements = Elements::default();

        let hydrogen = elements.push(Element {
            mass: 1.008,
            symbol: "H".into(),
        });
        let oxygen = elements.push(Element {
            mass: 15.999,
            symbol: "O".into(),
        });

        let compounds = vec![
            Compound::new("H₂", 1, 0.0, &[(hydrogen, 2)]),
            Compound::new("H⁻", -1, 132.282, &[(hydrogen, 1)]),
            Compound::new("H", 0, 203.278, &[(hydrogen, 1)]),
            //Compound::new("H₂⁺", 1, 1484.931, &[(hydrogen, 2)]),
            //Compound::new("H⁺", 1, 1516.990, &[(hydrogen, 1)]),

            Compound::new("O₂", 0, 0.0, &[(hydrogen, 2)]),
            Compound::new("O⁻", -1, 91.638, &[(hydrogen, 1)]),
            Compound::new("O", 0, 231.736, &[(hydrogen, 1)]),
            //Compound::new("O⁺²", 1, 1164.315, &[(hydrogen, 1)]),
            //Compound::new("O⁺", 1, 1546.912, &[(hydrogen, 1)]),

            Compound::new("OH⁻", -1, -138.698, &[(hydrogen, 1), (oxygen, 1)]),
            Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)]),
            Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)]),
            Compound::new("H₂O₂", 0, -105.445, &[(hydrogen, 2), (oxygen, 2)]),
        ];

        let chem = ChemicalWorld::from_laws(Laws {
            elements,
            compounds,
        });
        */

        //Self { chem }
        Self { }
    }
}

impl eframe::App for TemplateApp {
    /*
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.save_data);
    }
    */

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("hi");
        });
    }
}
