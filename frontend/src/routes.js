import ControlCenter from "layouts/control-center";
import uiText from "constants/uiText";

import { IoGrid } from "react-icons/io5";

const routes = [
  {
    type: "collapse",
    name: uiText.nav.sidebar,
    key: "control-center",
    route: "/",
    icon: <IoGrid size="15px" color="inherit" />,
    component: ControlCenter,
    noCollapse: true,
  },
];

export default routes;
