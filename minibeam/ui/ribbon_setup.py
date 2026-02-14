from __future__ import annotations

from ..common_ui.ribbon import (
    ActionRegistry,
    PyQtRibbonFactory,
    RibbonGroup,
    RibbonItem,
    RibbonSpec,
    RibbonTab,
)


def build_main_ribbon(mainwindow):
    spec = _build_spec(mainwindow)
    registry = _build_registry(mainwindow)
    return PyQtRibbonFactory().build(mainwindow, spec, registry)


def _build_registry(mainwindow) -> ActionRegistry:
    registry = ActionRegistry()
    for key in [
        "act_new", "act_open", "act_save", "act_undo", "act_redo", "act_delete",
        "act_select", "act_add_point", "act_materials", "act_sections", "act_assign_prop",
        "act_add_dx", "act_add_bush", "act_add_fy", "act_add_udl",
        "act_validate", "act_solve", "act_show_results", "act_export_csv", "act_export_report", "act_export_bundle",
        "act_fit_all", "act_reset_view",
        "act_bg_import", "act_bg_calibrate", "act_bg_opacity", "act_bg_bw", "act_bg_visible", "act_bg_clear",
        "act_help_pdf", "act_about",
    ]:
        registry.register(key, getattr(mainwindow, key))

    registry.register_widget("language_switch", lambda: mainwindow.cmb_language)

    return registry


def _build_spec(mainwindow) -> RibbonSpec:
    t = mainwindow._tr
    large = "L"
    return RibbonSpec(
        tabs=[
            RibbonTab(t("tab.home"), groups=[
                RibbonGroup(t("group.file"), items=[
                    RibbonItem("act_new", kind="action", size=large),
                    RibbonItem("act_open", kind="action", size=large),
                    RibbonItem("act_save", kind="action", size=large),
                ]),
                RibbonGroup(t("group.edit"), items=[
                    RibbonItem("act_undo", kind="action", size=large),
                    RibbonItem("act_redo", kind="action", size=large),
                    RibbonItem("act_delete", kind="action", size=large),
                ]),
                RibbonGroup(t("group.language"), items=[
                    RibbonItem("language_switch", kind="widget", size=large),
                ]),
            ]),
            RibbonTab(t("tab.model"), groups=[
                RibbonGroup(t("group.geometry"), items=[
                    RibbonItem("act_select", kind="action", size=large),
                    RibbonItem("act_add_point", kind="action", size=large),
                ]),
                RibbonGroup(t("group.properties"), items=[
                    RibbonItem("act_materials", kind="action", size=large),
                    RibbonItem("act_sections", kind="action", size=large),
                    RibbonItem("act_assign_prop", kind="action", size=large),
                ]),
            ]),
            RibbonTab(t("tab.boundary"), groups=[
                RibbonGroup(t("group.constraints"), items=[
                    RibbonItem("act_add_dx", kind="action", size=large, text_override="Add D"),
                    RibbonItem("act_add_bush", kind="action", size=large),
                ]),
                RibbonGroup(t("group.loads"), items=[
                    RibbonItem("act_add_fy", kind="action", size=large, text_override="Add F"),
                    RibbonItem("act_add_udl", kind="action", size=large),
                ]),
            ]),
            RibbonTab(t("tab.analysis"), groups=[
                RibbonGroup(t("group.solve"), items=[
                    RibbonItem("act_validate", kind="action", size=large),
                    RibbonItem("act_solve", kind="action", size=large),
                ]),
                RibbonGroup(t("group.results"), items=[
                    RibbonItem("act_show_results", kind="action", size=large),
                    RibbonItem("act_export_csv", kind="action", size=large),
                    RibbonItem("act_export_report", kind="action", size=large),
                    RibbonItem("act_export_bundle", kind="action", size=large),
                ]),
            ]),
            RibbonTab(t("tab.view"), groups=[
                RibbonGroup(t("group.navigation"), items=[
                    RibbonItem("act_fit_all", kind="action", size=large),
                    RibbonItem("act_reset_view", kind="action", size=large),
                ]),
            ]),
            RibbonTab(t("tab.background"), groups=[
                RibbonGroup(t("group.image"), items=[
                    RibbonItem("act_bg_import", kind="action", size=large),
                    RibbonItem("act_bg_calibrate", kind="action", size=large),
                    RibbonItem("act_bg_opacity", kind="action", size=large),
                    RibbonItem("act_bg_bw", kind="toggle", size=large),
                    RibbonItem("act_bg_visible", kind="toggle", size=large),
                    RibbonItem("act_bg_clear", kind="action", size=large),
                ]),
            ]),
            RibbonTab(t("tab.help"), groups=[
                RibbonGroup(t("group.support"), items=[
                    RibbonItem("act_help_pdf", kind="action", size=large),
                    RibbonItem("act_about", kind="action", size=large),
                ]),
            ]),
        ]
    )
