import Card from "@mui/material/Card";

import VuiBox from "components/VuiBox";
import VuiTypography from "components/VuiTypography";

function Footer() {
  return (
    <Card sx={{ mt: 4, p: "20px" }}>
      <VuiBox
        display="flex"
        flexDirection={{ xs: "column", md: "row" }}
        justifyContent="space-between"
        gap={2}
      >
        <VuiBox>
          <VuiTypography color="white" variant="lg" fontWeight="bold">
            Unturned
          </VuiTypography>
          <VuiTypography color="text" variant="caption" display="block" mt={1}>
            Lenta Tech Life Hack
          </VuiTypography>
        </VuiBox>

        <VuiBox textAlign={{ xs: "left", md: "right" }}>
          <VuiTypography
            component="a"
            href="https://github.com/n3onnhowever/unturned-lenta-tech"
            target="_blank"
            rel="noreferrer"
            color="info"
            variant="button"
            display="block"
          >
            GitHub-репозиторий
          </VuiTypography>
          <VuiTypography color="text" variant="caption" display="block" mt={1}>
            © 2026 Unturned Team
          </VuiTypography>
        </VuiBox>
      </VuiBox>
    </Card>
  );
}

export default Footer;
