/*!

=========================================================
* Vision UI Free React - v1.0.0
=========================================================

* Product Page: https://www.creative-tim.com/product/vision-ui-free-react
* Copyright 2021 Creative Tim (https://www.creative-tim.com/)
* Licensed under MIT (https://github.com/creativetimofficial/vision-ui-free-react/blob/master LICENSE.md)

* Design and Coded by Simmmple & Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/

// Vision UI Dashboard React Base Styles
import colors from "assets/theme/base/colors";

const { info, dark } = colors;
export default {
  html: {
    scrollBehavior: "smooth",
    background: dark.body,
  },
  body: {
    background:
      "radial-gradient(circle at 28% 18%, rgba(0, 117, 255, 0.34), transparent 30%), radial-gradient(circle at 76% 78%, rgba(33, 150, 255, 0.24), transparent 34%), linear-gradient(135deg, #020916 0%, #061b3d 46%, #031026 100%)",
    backgroundSize: "cover",
    backgroundAttachment: "fixed",
  },
  "*, *::before, *::after": {
    margin: 0,
    padding: 0,
  },
  "a, a:link, a:visited": {
    textDecoration: "none !important",
  },
  "a.link, .link, a.link:link, .link:link, a.link:visited, .link:visited": {
    color: `${dark.main} !important`,
    transition: "color 150ms ease-in !important",
  },
  "a.link:hover, .link:hover, a.link:focus, .link:focus": {
    color: `${info.main} !important`,
  },
};
