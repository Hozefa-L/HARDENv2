contract AgencyRegistryV1 {
    address public administrator;
    bool public initialized;
    // SWC-135: no constructor, no init guard
    function initialize(address admin) external {
        administrator = admin;
        initialized = true;
    }
    function setRecord(uint256 id, bytes32 data) external {
        require(msg.sender == administrator);
        // ... record update elided
    }
}