import { useState, useEffect } from "react";
import PropTypes from "prop-types";

import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import Icon from "@mui/material/Icon";

import VuiBox from "components/VuiBox";
import Breadcrumbs from "examples/Breadcrumbs";
import uiText from "constants/uiText";

import {
  navbar,
  navbarContainer,
  navbarRow,
  navbarMobileMenu,
} from "examples/Navbars/DashboardNavbar/styles";

import { useVisionUIController, setTransparentNavbar, setMiniSidenav } from "context";

function DashboardNavbar({ absolute, light, isMini }) {
  const [navbarType, setNavbarType] = useState();
  const [controller, dispatch] = useVisionUIController();
  const { miniSidenav, transparentNavbar, fixedNavbar } = controller;

  useEffect(() => {
    if (fixedNavbar) {
      setNavbarType("sticky");
    } else {
      setNavbarType("static");
    }

    function handleTransparentNavbar() {
      setTransparentNavbar(dispatch, (fixedNavbar && window.scrollY === 0) || !fixedNavbar);
    }

    window.addEventListener("scroll", handleTransparentNavbar);
    handleTransparentNavbar();

    return () => window.removeEventListener("scroll", handleTransparentNavbar);
  }, [dispatch, fixedNavbar]);

  const handleMiniSidenav = () => setMiniSidenav(dispatch, !miniSidenav);

  return (
    <AppBar
      position={absolute ? "absolute" : navbarType}
      color="inherit"
      sx={(theme) => navbar(theme, { transparentNavbar, absolute, light })}
    >
      <Toolbar sx={(theme) => navbarContainer(theme)}>
        <VuiBox color="inherit" mb={{ xs: 1, md: 0 }} sx={(theme) => navbarRow(theme, { isMini })}>
          <Breadcrumbs icon="home" title={uiText.nav.breadcrumbTitle} route={[uiText.nav.breadcrumbRoute]} light={light} />
        </VuiBox>
        {isMini ? null : (
          <VuiBox sx={(theme) => navbarRow(theme, { isMini })}>
            <VuiBox color={light ? "white" : "inherit"}>
              <IconButton
                size="small"
                color="inherit"
                sx={navbarMobileMenu}
                onClick={handleMiniSidenav}
              >
                <Icon className="text-white">{miniSidenav ? "menu_open" : "menu"}</Icon>
              </IconButton>
            </VuiBox>
          </VuiBox>
        )}
      </Toolbar>
    </AppBar>
  );
}

DashboardNavbar.defaultProps = {
  absolute: false,
  light: false,
  isMini: false,
};

DashboardNavbar.propTypes = {
  absolute: PropTypes.bool,
  light: PropTypes.bool,
  isMini: PropTypes.bool,
};

export default DashboardNavbar;
