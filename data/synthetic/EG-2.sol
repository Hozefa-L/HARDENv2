contract PermittedVoteTally {
    address public clerk;
    mapping(uint16 => uint8) public tally; // SWC-101 hazard
    constructor() { clerk = msg.sender; }
    function castBatch(uint16 precinct, uint8 votes) external {
        require(msg.sender == clerk, "auth");
        unchecked { tally[precinct] += votes; } // wraps silently
    }
}